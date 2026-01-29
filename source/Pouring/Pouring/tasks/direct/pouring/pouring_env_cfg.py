# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Custom imports
from .fluid_object import FluidObjectCfg, FluidObject
from pxr import Gf
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import RigidObjectCfg
import os
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.controllers import DifferentialIKControllerCfg


@configclass
class PouringEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - spaces definition
    action_space = 6
    observation_space = 19
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # path
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12, solver_velocity_iteration_count=1
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -1.4460,
                "panda_joint2": 0.2157,
                "panda_joint3": 1.2273,
                "panda_joint4": -2.4090,
                "panda_joint5": 2.8540,
                "panda_joint6": 2.2554,
                "panda_joint7": 0.7622, 
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0),
            rot=(0.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit=870.0,
                velocity_limit=2.175,
                stiffness=8.0e2,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit=120.0,
                velocity_limit=2.61,
                stiffness=8.0e2,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    # Joint names to actuate along the arm
    robot_arm_names = list()
    for i in range(1,8):
        robot_arm_names.append("panda_joint%d"%i)

    # Joint names of the fingers
    robot_finger_names = ["panda_finger_joint.*"]

    # Controller configuration
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=4.0, replicate_physics=False)


    # Set Glass as rigid object
    spawn_pos_glass = Gf.Vec3f(0.61, -0.1, 0.25)
    glass = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Glass",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_glass, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/Tall_Glass_5.usd",
            semantic_tags=[("class","Glass")],
            scale=(0.01, 0.01, 0.01),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # Set target container as rigid object
    spawn_pos_container = Gf.Vec3f(0.61, 0., 0.01)
    container = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Container",
        init_state=RigidObjectCfg.InitialStateCfg(pos=spawn_pos_container, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{CURRENT_PATH}/usd_models/Container.usd",
            semantic_tags=[("class","Container")],
            scale=(0.001, 0.001, 0.001),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # fluid object
    spawn_pos_fluid = Gf.Vec3f(0.0, 0.0, 4.0)  # Lower position for the spawn

    # Add liquid configuration parameters
    # Direct spawn
    liquidCfg = FluidObjectCfg()
    liquidCfg.numParticlesX = 8
    liquidCfg.numParticlesY = 8
    liquidCfg.numParticlesZ = 44
    liquidCfg.density = 0.0
    liquidCfg.particle_mass = 0.001
    liquidCfg.particleSpacing = 0.005
    liquidCfg.viscosity = 0.91

    # Spawn position of the center of the base of the fluid
    spawn_pos_fluid = spawn_pos_glass + Gf.Vec3f(0.0,0,0.05)

    # Fill levels inside the source container
    particles_init_pos_list = ["particle_init_pos_high"]

    # reward scales
    inside_weight = 1.0
    outside_weight = -1.0
    source_pos_weight = 0.
    source_ground_weight = -0.
    source_vel_weight = -0.00
    joint_vel_weight = 0.
    actions_weight = -0.1

    # Action scales
    action_scale_lin = 0.1
    action_scale_rot = 0.1
