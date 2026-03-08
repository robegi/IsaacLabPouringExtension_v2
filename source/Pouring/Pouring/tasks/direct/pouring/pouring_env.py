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
from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, quat_mul
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg, save_images_to_file
import omni.replicator.core as rep
import os

class PouringEnv(DirectRLEnv):
    cfg: PouringEnvCfg

    def __init__(self, cfg: PouringEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # create auxiliary variables for computing applied action, observations and rewards
        self.robot_joint_lower_limits = self._robot.data.soft_joint_pos_limits[0, :7, 0].to(device=self.device)
        self.robot_joint_upper_limits = self._robot.data.soft_joint_pos_limits[0, :7, 1].to(device=self.device)
        self.robot_joint_velocity_scale = 0.1
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

        # Glass, position it before the robot
        self._glass = RigidObject(self.cfg.glass)
        self.scene.rigid_objects["glass"] = self._glass

        # Container
        self._container = RigidObject(self.cfg.container)
        self.scene.rigid_objects["container"] = self._container

        # Container data from original usd model (to compute particles outside)
        self.container_height = 0.12
        self.container_radius = 0.15/2
        self.container_base_thickness = 0.02

        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Camera
        self._camera = TiledCamera(self.cfg.camera) 
        self.data_type = 'rgb'
        self.scene.sensors["camera"] = self._camera 

        # Create replicator writer
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
        self.rep_writer = rep.BasicWriter(
            output_dir=self.output_dir,
            frame_padding=0,
            colorize_instance_id_segmentation=self._camera.cfg.colorize_instance_id_segmentation,
            colorize_instance_segmentation=self._camera.cfg.colorize_instance_segmentation,
            colorize_semantic_segmentation=self._camera.cfg.colorize_semantic_segmentation,
        )

        # Controller
        self.diff_ik_controller = DifferentialIKController(self.cfg.diff_ik_cfg, num_envs=self.num_envs, device=self.device)
        self.first_setup = True
        self.ee_goal_start = torch.tensor([0.5, -0.1, 0.3, 0.707, 0, 0.707, 0], device = self.device)
        self.ee_goal = self.ee_goal_start.clone().expand(self.num_envs, -1)
        self.delta_pos = torch.zeros((self.num_envs, 3), device = self.device)
        self.delta_rot = torch.tensor([0], device = self.device).expand(self.num_envs, -1)
        self.delta_pos_limit_low = torch.tensor([0.3, -0.5, 0], device = self.device)
        self.delta_pos_limit_up = torch.tensor([0.8, 0.5, 0.5], device = self.device)

        # Liquid
        self.cfg.liquidCfg.num_envs = self.num_envs
        self.liquid = FluidObject(cfg=self.cfg.liquidCfg, lower_pos=self.cfg.spawn_pos_fluid)
        self.liquid.spawn_fluid()  # Spawn only in env 0, it is replicated automatically

        # # Initial particle position, from saved file
        self.liquid_init_pos = list()
        self.liquid_init_vel = list()

        for i in range(len(self.cfg.particles_init_pos_list)):
            self.liquid_init_pos.append(torch.load(f"{self.cfg.CURRENT_PATH}/usd_models/{self.cfg.particles_init_pos_list[i]}.pt").cuda())
            self.liquid_init_pos[i] += torch.ones_like(self.liquid_init_pos[i], device=self.device)*torch.tensor([0, 0, 0.01], device=self.device)
            self.liquid_init_vel.append(torch.zeros_like(self.liquid_init_pos[i], device=self.device))

        # Particles
        self.particle_pos = torch.ones((self.num_envs, self.liquid.particles_num, 3)).to(self.device)

        # Target on the finger actuators to hold the glass
        self.ee_finger_start = torch.tensor([0.5, 0.5], device=self.device).unsqueeze(0) # Initial finger position for resetting
        self.ee_target = torch.zeros((self.num_envs, 2), device = self.device)  

        # Initial joint target is the starting position
        self.robot_dof_targets = torch.tensor(list(self._robot.cfg.init_state.joint_pos.values()), device=self.device)

        # Initialize variables to store useful quantities
        self.spilled_fraction = torch.zeros((self.num_envs, self.liquid.particles_num, 3), device = self.device)
        self.inside_fraction = torch.zeros((self.num_envs, self.liquid.particles_num, 3), device = self.device)
        self.container_pos = torch.zeros((self.num_envs, 3), device = self.device)
        self.obs = {"camera": torch.zeros((self.num_envs, self.cfg.num_channels, self.cfg.camera.width, self.cfg.camera.height), device = self.device), 
                    "sensors": torch.zeros((self.num_envs, self.cfg.num_sensors), device = self.device)}


        # Marker on the end effector and the desired pose
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # Auxiliary variables for testing
        self.counter = 0


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1, 1)
        self.delta_pos = self.actions[:, :3]*self.cfg.action_scale_lin
        self.delta_rot = self.actions[:, 3]*self.cfg.action_scale_rot

        # # ----------------------------------------------------------------------------------------------------------------------
        # # Testing actions (comment out)
        # if self.counter == (120/2*self.num_envs)/2:
        #     self.delta_pos= torch.tensor([0, -0.1, 0]).cuda()

        # elif self.counter == (120/2*self.num_envs)*3:
        #     self.delta_pos= torch.tensor([0, 0, -0.]).cuda()

        # elif self.counter == (120/2*self.num_envs)*2:
        #     self.delta_rot= torch.tensor([-math.pi/2]).expand(self.num_envs, -1).cuda()

        # else:
        #     self.delta_pos= torch.tensor([0, 0, 0]).cuda()
        #     self.delta_rot= torch.tensor([0]).expand(self.num_envs, -1).cuda()
            

        # self.counter += 1
        # # -----------------------------------------------------------------------------------------------------------------------
        
        # Apply actions
        ee_pos = self.ee_goal[:,:3] + self.delta_pos
        ee_pos = ee_pos.clamp(self.delta_pos_limit_low, self.delta_pos_limit_up)
        ee_rot = quat_mul(self.ee_goal[:, 3:], quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), self.delta_rot))
        self.ee_goal = torch.cat((ee_pos, ee_rot), dim=1) 
        
        ###
        # IK controller (target)
        ###
        ee_pose_w = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(self.ee_goal, ee_pos_b, ee_quat_b)    

        # Markers
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.ee_goal[:, 0:3] + self.scene.env_origins[:, 0:3], self.ee_goal[:, 3:7])
    


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

        # Camera
        
        # Extract and save rgb output from camera
        camera_data = self._camera.data.output[self.data_type]/255.0
        # Choose whether to save the images or not
        images_are_being_saved = True
        
        if images_are_being_saved:
            self.save_image(camera_data, self.index_image, 0, "rgb")

        self.index_image +=1 # Index for saving the images
        # Subtract the mean from the camera input
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor
        self.obs["camera"] = camera_data

        # Sensors

        # Get particles data
        self.particle_pos=self.liquid.get_particles_position()
        
        # Scaled joint positions (exclude fingers)
        joint_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos[:,:7] - self.robot_joint_lower_limits)
            / (self.robot_joint_upper_limits - self.robot_joint_lower_limits)
            - 1.0
        )

        # Scaled joint velocities
        joint_vel = self._robot.data.joint_vel[:,:7] * self.robot_joint_velocity_scale

        # EE position
        ee_pose_w = self._robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        
        # Container position in plane (local frame)
        self.container_pos = self._container.data.root_pos_w[:,:2] - self.scene.env_origins[:,:2]

        # In/out fraction
        self.spilled_fraction, self.inside_fraction = self.get_particles_in_out_fraction(particles_pos=self.particle_pos,
                                                            container_pos = self.container_pos,
                                                            container_base = self.container_base_thickness,
                                                            container_height = self.container_height,
                                                            container_radius = self.container_radius,
                                                            total_particles = self.liquid.particles_num)
        
        # Concatenate observations
        self.obs["sensors"] = torch.cat(
            (
                joint_pos_scaled,
                joint_vel,
                ee_pos_b,
                ee_quat_b,
                self.ee_goal,
                self.container_pos,
                self.spilled_fraction,
                self.inside_fraction,
            ),
            dim=-1,
        )
        observations = {"policy": self.obs}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:

        # Penalty for velocity magnitude
        joint_vel = self._robot.data.joint_vel[:,:7]
        reward_actions = torch.sum(joint_vel**2, dim=-1)
        reward_actions= self.cfg.actions_weight*reward_actions.unsqueeze(1)

        reward_inside = self.inside_fraction*self.cfg.inside_weight
        reward_outside = self.spilled_fraction*self.cfg.outside_weight

        # Penalty for source container below threshold
        source_pos = self._glass.data.root_pos_w[:,:3] - self.scene.env_origins
        reward_source_pos = torch.where(source_pos[:,2]<self.container_height + 0.03, 1.0, .0).unsqueeze(1)*self.cfg.source_pos_weight

        total_reward = reward_inside + reward_outside + reward_actions + reward_source_pos

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset if most fluid is poured outside 
        # In/out fraction
        spilled_fraction, inside_fraction = self.get_particles_in_out_fraction(particles_pos=self.particle_pos,
                                                            container_pos = self.container_pos,
                                                            container_base = self.container_base_thickness,
                                                            container_height = self.container_height,
                                                            container_radius = self.container_radius,
                                                            total_particles = self.liquid.particles_num)
        terminated = torch.any(spilled_fraction > 0.5, dim = 1) # Reset if most liquid poured outside
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
        self.ee_goal[env_ids] = self.ee_goal_start.clone()
        self.delta_pos = torch.zeros((self.num_envs, 3), device = self.device)
        self.delta_rot = torch.tensor([0], device = self.device).expand(self.num_envs, -1)
        
        # Reset the robot 
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos = torch.clamp(joint_pos[:,:7], self.robot_joint_lower_limits, self.robot_joint_upper_limits)
        joint_pos = torch.cat((joint_pos, self.ee_finger_start.expand([len(env_ids), -1])), dim=1)
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

        # Resets fluid
        self.liquid.set_particles_position_and_velocity(env_ids = env_ids, particles_pos=self.liquid_init_pos[0], particles_vel=self.liquid_init_vel[0])

        # Reset camera saving index
        self.index_image = 0

        # Testing variables reset
        self.counter = 0

    ###
    # Auxiliary methods
    ###
    @torch.jit.script
    def get_particles_in_out_fraction(particles_pos: torch.Tensor,
                                       container_pos: torch.Tensor,
                                       container_height: float,
                                       container_radius: float,
                                       container_base: float,
                                       total_particles: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Computes the fraction of particles that are outside the container and below a certain height
        height_condition = particles_pos[:, :, 2] < container_height + container_base
        x = particles_pos[:, :, 0] - container_pos[:, 0].unsqueeze(1)
        y = particles_pos[:, :, 1] - container_pos[:, 1].unsqueeze(1)
        outside_condition = x**2 + y**2 >= container_radius**2
        particles_outside = torch.sum(torch.where(height_condition & outside_condition, 1, 0), dim = 1)
        inside_condition = x**2 + y**2 < container_radius**2
        particles_inside = torch.sum(torch.where(height_condition & inside_condition, 1, 0), dim = 1)

        fraction_outside = particles_outside / total_particles
        fraction_outside = fraction_outside.reshape(-1, 1)

        fraction_inside = particles_inside / total_particles
        fraction_inside = fraction_inside.reshape(-1, 1)

        return fraction_outside, fraction_inside

    @torch.jit.script
    def get_particles_inside_fraction(particles_pos: torch.Tensor,
                                       container_pos: torch.Tensor,
                                       container_height: float,
                                       container_radius: float,
                                       container_base: float,
                                       total_particles: int) -> torch.Tensor:
        
        # Computes the fraction of particles that are outside the container and below a certain height
        height_condition = particles_pos[:, :, 2] < container_height + container_base
        x = particles_pos[:, :, 0] - container_pos[:, 0].unsqueeze(1)
        y = particles_pos[:, :, 1] - container_pos[:, 1].unsqueeze(1)
        inside_condition = x**2 + y**2 < container_radius**2
        particles_inside = torch.sum(torch.where(height_condition & inside_condition, 1, 0), dim = 1)
        
        fraction_inside = particles_inside / total_particles

        return fraction_inside.reshape(-1, 1)
    
    def save_image(self, file, index_image, index_env, name):
        # Save images from camera 
        if not torch.is_tensor(file):
            file = torch.tensor(file, device=self.device)
        # Adjust dimensions
        if len(file.shape)<4:
            file = torch.unsqueeze(file, 0)
        # Expand number of channels
        if file.shape[3]==1:
            #print(file.unique())
            file_new = torch.zeros((file.shape[0],file.shape[1],file.shape[2],3), device=self.device)
            file_new[:] = file 
            file = file_new
            #print(file.unique())
        save_images_to_file(file, f"{self.cfg.CURRENT_PATH}/output/camera/{name}_{index_env}_{index_image}.png")

    