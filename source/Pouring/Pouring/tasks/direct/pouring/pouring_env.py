# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .pouring_env_cfg import PouringEnvCfg

# Custom imports
from .fluid_object import FluidObject
from omni.physx import acquire_physx_interface
import carb
from isaaclab.assets import RigidObject
from isaaclab.controllers import DifferentialIKController
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers.config import FRAME_MARKER_CFG

class PouringEnv(DirectRLEnv):
    cfg: PouringEnvCfg

    def __init__(self, cfg: PouringEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1].to(device=self.device)
        self.robot_dof_speed_scales = torch.ones((self.num_envs, self._robot.num_joints - 2), device=self.device)
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # Initial joint target is the starting position
        self.robot_dof_targets = torch.tensor(list(self.cfg.robot.init_state.joint_pos.values()), device=self.device)[:7]


        self._robot_arm_idx, _ = self._robot.find_joints(self.cfg.robot_arm_names)
        self._robot_finger_idx = self._robot.find_joints(self.cfg.robot_finger_names)

        self.action_constraints_low = torch.tensor([-0.3, -math.pi], device=self.device)
        self.action_constraints_high = torch.tensor([0.3, 0], device=self.device)
        


    def _setup_scene(self):

        # Use device without forcing anything
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(-1)

        # Set translucency to render transparent materials
        settings = carb.settings.get_settings()
        settings.set("/rtx/translucency/enabled", True)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # Need to explicitly filter collisions after cloning envs
        self.scene.filter_collisions(global_prim_paths=[])
        # add lights 
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Controller
        self.diff_ik_controller = DifferentialIKController(self.cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.first_setup = True

        # Liquid
        self.liquid = FluidObject(cfg=self.cfg.liquidCfg, lower_pos=self.cfg.spawn_pos_fluid)
        self.liquid.spawn_fluid_direct()

        # # Initial particle position, from saved file
        self.liquid_init_pos = list()
        self.liquid_init_vel = list()

        for i in range(len(self.cfg.particles_init_pos_list)):
            self.liquid_init_pos.append(torch.load(f"{self.cfg.CURRENT_PATH}/usd_models/{self.cfg.particles_init_pos_list[i]}.pt").cuda())
            self.liquid_init_pos[i] += torch.ones_like(self.liquid_init_pos[i], device=self.device)*torch.tensor([0, 0, 0.01], device=self.device)
            self.liquid_init_vel.append(torch.zeros_like(self.liquid_init_pos[i], device=self.device))

        # Reward and observations
        self.reward = torch.zeros((self.num_envs)).to(self.device)
        self.obs_reward_in = torch.zeros((self.num_envs)).to(self.device)
        self.obs_reward_out = torch.zeros((self.num_envs)).to(self.device)
        self.particle_fraction_in = torch.zeros((self.num_envs,1)).to(self.device)
        self.particle_fraction_out = torch.zeros((self.num_envs,1)).to(self.device)
        
        # Glass, position it before the robot
        self._glass = RigidObject(self.cfg.glass)
        self.scene.rigid_objects["glass"] = self._glass

        # Container
        self._container = RigidObject(self.cfg.container)
        self.scene.rigid_objects["container"] = self._container

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Target on the finger actuators to hold the glass
        self.ee_start = torch.tensor([0.4, 0.4], device=self.device).unsqueeze(0) # Initial finger position for resetting
        self.ee_target = torch.zeros((self.num_envs, 2), device = self.device)  

        # Initial joint target is the starting position
        self.robot_dof_targets = torch.tensor(list(self._robot.cfg.init_state.joint_pos.values()), device=self.device)

        # Marker on the end effector and the desired pose
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-0.01, 0.01)
        ee_goal = self.actions
        
        ###
        # IK controller (target)
        ###
        ee_pose_w = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(ee_goal, ee_pos_b, ee_quat_b)    

        # Markers
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
    


    def _apply_action(self) -> None:
        ###
        # IK controller (step)
        ###
        # Get jacobian and current pose
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids].to(self.device)
        ee_pose_w = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w
        joint_pos = self._robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # compute the joint commands
        self.joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        self._robot.set_joint_position_target(self.joint_pos_des, joint_ids=self._robot_arm_idx)
        self._robot.set_joint_position_target(self.ee_target, joint_ids=self._robot_finger_idx[0])
        

    def _get_observations(self) -> dict:
        obs = torch.ones((self.num_envs, self.cfg.observation_space), device=self.device)
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = torch.zeros((self.num_envs), device=self.device).unsqueeze(1)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = False
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # IK controller
        if self.first_setup == True:
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            self.robot_entity_cfg.resolve(self.scene)
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1  # -1 because base link is counted
            self.first_setup = False

        # Reset controller
        self.diff_ik_controller.reset()
        
        # Reset the robot and randomizes the initial position (to implement)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos = torch.clamp(joint_pos[:,:7], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_pos = torch.cat((joint_pos, self.ee_start.expand([self.num_envs, -1])), dim=1)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

         # Reset the glass
        glass_init_pos = self._glass.data.default_root_state.clone()[env_ids]
        glass_init_pos[:,:3] = glass_init_pos[:,:3] + self.scene.env_origins[env_ids]
        self._glass.write_root_state_to_sim(glass_init_pos,env_ids=env_ids)

        # Reset the container
        container_init_pos = self._container.data.default_root_state.clone()[env_ids]
        container_init_pos[:,:3] = container_init_pos[:,:3] + self.scene.env_origins[env_ids]
        lower_bound = torch.tensor([0,-0.1,0],device=self.device)
        upper_bound = torch.tensor([0,0.1,0],device=self.device)
        container_init_pos[:,:3] += sample_uniform(lower_bound, upper_bound, container_init_pos[:,:3].shape, self.device) # Randomize
        self._container.write_root_state_to_sim(container_init_pos,env_ids=env_ids)

        # # Resets fluid
        for i in env_ids:
            self.liquid.set_particles_position(env_id = i)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward