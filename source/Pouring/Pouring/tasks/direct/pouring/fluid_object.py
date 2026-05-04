from omni.physx.scripts import physicsUtils, particleUtils, utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema, UsdShade
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
import torch
from typing import Union


class FluidObjectCfg():

    # Number of particles along the horizontal and vertical axes (for direct spawn)
    numParticlesX: int = 5
    numParticlesY: int = 5
    numParticlesZ: int = 5

    # Cylinder dimension (for sampled spawn)
    radius: float = 0.1
    height: float = 0.4

    # Fluid properties
    particle_mass : float = 0.001
    density : float = 0.0
    viscosity: float = 0.91
    particleSpacing: float = 0.005

    # Environment parameters
    num_envs: int = 1

     

class FluidObject():

    cfg: FluidObjectCfg

    def __init__(self, cfg: FluidObjectCfg, pos: Gf.Vec3f):
        self.cfg = cfg
        self.lower_pos = pos # Lower position for the spawn
        
        # Scene infos (Default values)
        context = omni.usd.get_context()
        self.stage = context.get_stage()
        self.default_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World")).GetPrim()
        self.stage.SetDefaultPrim(self.default_prim)
        self.default_prim_path = self.stage.GetDefaultPrim().GetPath()
        self.scenePath = Sdf.Path("/physicsScene")

        # Initialize internal variables 
        self.initial_particles_pos: torch.Tensor | None = None
        self.initial_particles_vel: torch.Tensor | None = None
        self.num_particles: int | None = None


    def spawn_fluid_direct(self, env_id: int = 0):
            ###
            # Spawns the fluid particles (in the environment 0 by default)
            ###
            
            # Particle System
            self.particleSystemPath = self.default_prim_path.AppendChild("particleSystem")

            # Particle points
            self.particlesPath = Sdf.Path(f"/World/envs/env_{env_id}/particles")

            # solver iterations
            self._solverPositionIterations = 4
            physxAPI = PhysxSchema.PhysxSceneAPI.Apply(self.stage.GetPrimAtPath(self.scenePath))
            physxAPI.CreateSolverTypeAttr("TGS")

            # particle params
            restOffset = self.cfg.particleSpacing * 0.9
            fluidRestOffset = restOffset * 0.6
            particleContactOffset = restOffset + 0.001
            particle_system = particleUtils.add_physx_particle_system(
                stage=self.stage,
                particle_system_path=self.particleSystemPath,
                simulation_owner=self.scenePath,
                contact_offset=restOffset * 1.5 + 0.01,
                rest_offset=restOffset * 1.5,
                particle_contact_offset=particleContactOffset,
                solid_rest_offset=0.0,
                fluid_rest_offset=fluidRestOffset,
                solver_position_iterations=self._solverPositionIterations,
            )

            mtl_created = []
            omni.kit.commands.execute(
                "CreateAndBindMdlMaterialFromLibrary",
                mdl_name="OmniSurfacePresets.mdl",
                mtl_name="OmniSurface_DeepWater",
                mtl_created_list=mtl_created,
            )
            pbd_particle_material_path = mtl_created[0]
            omni.kit.commands.execute(
                "BindMaterial", prim_path=self.particleSystemPath, material_path=pbd_particle_material_path
            )

            # Create a pbd particle material and set it on the particle system
            particleUtils.add_pbd_particle_material(
                self.stage,
                pbd_particle_material_path,
                cohesion=10,
                viscosity=self.cfg.viscosity,
                surface_tension=0.74,
                friction=0.1,
            )
            physicsUtils.add_physics_material_to_prim(self.stage, particle_system.GetPrim(), pbd_particle_material_path)

            particle_system.CreateMaxVelocityAttr().Set(200)

            # # add particle anisotropy
            # anisotropyAPI = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
            # anisotropyAPI.CreateParticleAnisotropyEnabledAttr().Set(True)
            # aniso_scale = 5.0
            # anisotropyAPI.CreateScaleAttr().Set(aniso_scale)
            # anisotropyAPI.CreateMinAttr().Set(1.0)
            # anisotropyAPI.CreateMaxAttr().Set(2.0)

            # # add particle smoothing
            # smoothingAPI = PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())
            # smoothingAPI.CreateParticleSmoothingEnabledAttr().Set(True)
            # smoothingAPI.CreateStrengthAttr().Set(0.5)

            # # apply isosurface params
            # isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
            # isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
            # isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
            # isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
            # isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
            # isosurfaceAPI.CreateGridSpacingAttr().Set(fluidRestOffset * 1.5)
            # isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluidRestOffset * 1.6)
            # isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
            # isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluidRestOffset * 2)

            # isosurfaceAPI.CreateNumMeshSmoothingPassesAttr().Set(1)

            # primVarsApi = UsdGeom.PrimvarsAPI(particle_system)
            # primVarsApi.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool).Set(True)

            # self.stage.SetInterpolationType(Usd.InterpolationTypeHeld)

            # Create Grid
            gridSpacing = self.cfg.particleSpacing + 0.001
            lower = self.lower_pos + Gf.Vec3f(-gridSpacing*self.cfg.numParticlesX/2, -gridSpacing*self.cfg.numParticlesY/2, 0) # Translate lower corner
            positions, velocities = particleUtils.create_particles_grid(
                lower, gridSpacing, self.cfg.numParticlesX, self.cfg.numParticlesY, self.cfg.numParticlesZ
            )


            widths = [self.cfg.particleSpacing] * len(positions)
            
            self.particlesPrim = particleUtils.add_physx_particleset_points(
                stage=self.stage,
                path=self.particlesPath,
                positions_list=Vt.Vec3fArray(positions),
                velocities_list=Vt.Vec3fArray(velocities),
                widths_list=widths,
                particle_system_path=self.particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg.particle_mass,
                density=self.cfg.density,
            )

            # # Hide particles
            # visibility_attribute = self.particlesPrim.GetVisibilityAttr()
            # visibility_attribute.Set("invisible")

            # Saves particle data
            self.initial_particles_pos = self.get_particles_position([0])
            self.initial_particles_vel = self.get_particles_velocity([0])
            self.num_particles = self.cfg.numParticlesX * self.cfg.numParticlesY * self.cfg.numParticlesZ

    def spawn_fluid_sampler(self, env_id: int = 0):
        ###
        # Spawns fluid particles by sampling a mesh (in environment 0 by default)
        # Code reference in the isaac sim particle sampler demo
        ###

        # Empty internal variables
        self.initial_particles_pos = None
        self.initial_particles_vel = None
        self.num_particles = None

        # configure and create particle system
        particle_system_path = self.default_prim_path.AppendChild("particleSystem")
        particle_system = PhysxSchema.PhysxParticleSystem.Define(self.stage, particle_system_path)
        particle_system.CreateSimulationOwnerRel().SetTargets([self.scenePath])
        # The simulation determines the other offsets from the particle contact offset
        Particle_Contact_Offset = self.cfg.particleSpacing # From cfg data
        particle_system.CreateParticleContactOffsetAttr().Set(Particle_Contact_Offset)
        # Limit particle velocity for better collision detection
        particle_system.CreateMaxVelocityAttr().Set(250.0)

        # create particle material and assign it to the system:
        particle_material_path = self.default_prim_path.AppendChild("particleMaterial")
        particleUtils.add_pbd_particle_material(self.stage, particle_material_path)
        physicsUtils.add_physics_material_to_prim(
            self.stage, self.stage.GetPrimAtPath(particle_system_path), particle_material_path
        )

        # create a cylinder mesh that shall be sampled:
        # cylinder_mesh_path = Sdf.Path(omni.usd.get_stage_next_free_path(self.stage, "/Cylinder", True))
        cylinder_mesh_path = Sdf.Path(f"/World/envs/env_{0}/Cylinder")
        cylinder_resolution = (
            10  # resolution can be low because we'll sample the surface / volume only irrespective of the vertex count
        )
        omni.kit.commands.execute(
            "CreateMeshPrim", prim_type="Cylinder", u_patches=cylinder_resolution, v_patches=cylinder_resolution, select_new_prim=False,
            prim_path = cylinder_mesh_path
        )
        cylinder_mesh = UsdGeom.Mesh.Get(self.stage, cylinder_mesh_path)
        physicsUtils.set_or_add_scale_op(cylinder_mesh, Gf.Vec3f(self.cfg.radius, self.cfg.radius, self.cfg.height))
        physicsUtils.set_or_add_translate_op(cylinder_mesh, self.lower_pos) # Translate to the spawn position

        # configure target particle set:
        # particle_points_path = self.default_prim_path.AppendChild("sampledParticles")
        particle_points_path = Sdf.Path(f"/World/envs/env_{env_id}/particles")
        points = UsdGeom.Points.Define(self.stage, particle_points_path)
        # add render material:
        material_path = self.create_pbd_material("OmniPBR")
        omni.kit.commands.execute(
            "BindMaterialCommand", prim_path=particle_points_path, material_path=material_path, strength=None
        )

        particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
        PhysxSchema.PhysxParticleAPI(particle_set_api).CreateParticleSystemRel().SetTargets([particle_system_path])

        # compute particle sampler sampling distance
        # use particle fluid restoffset to determine sampler distance, using same formula as simulation, see
        # https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_physics.html#offset-autocomputation
        fluid_rest_offset = 0.99 * 0.6 * Particle_Contact_Offset
        particle_sampler_distance = 2.0 * fluid_rest_offset

        # reference the particle set in the sampling api
        sampling_api = PhysxSchema.PhysxParticleSamplingAPI.Apply(cylinder_mesh.GetPrim())
        sampling_api.CreateParticlesRel().AddTarget(particle_points_path)
        sampling_api.CreateSamplingDistanceAttr().Set(particle_sampler_distance)
        sampling_api.CreateMaxSamplesAttr().Set(5e5)
        sampling_api.CreateVolumeAttr().Set(True)

        # apply isosurface params
        isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
        isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
        isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
        isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
        isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
        isosurfaceAPI.CreateGridSpacingAttr().Set(fluid_rest_offset * 1.5)
        isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluid_rest_offset * 1.6)
        isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
        isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluid_rest_offset * 2)

    def get_particles_position(self, env_ids: Union[list[int], None] = None) -> torch.Tensor:
        # Gets particles' positions in the input environment and velocities and outputs them as torch tensors
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        particles_pos = torch.zeros((len(env_ids), self.num_particles, 3), device='cuda')
        for i in env_ids:
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{i}/particles")))
            particles_pos[i] = torch.from_numpy(np.asarray(particles.GetPointsAttr().Get())).cuda()

        return particles_pos
    
    def get_particles_velocity(self, env_ids: Union[list[int], None] = None) -> torch.Tensor:
        # Gets particles' velocities in the input environment and outputs them as torch tensors
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        particles_vel = torch.zeros((len(env_ids), self.num_particles, 3), device='cuda')
        # Cycle through all environments
        for i in env_ids:
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{i}/particles")))
            particles_vel[i] = torch.from_numpy(np.asarray(particles.GetVelocitiesAttr().Get())).cuda()

        return particles_vel

    def set_particles_position_and_velocity(self, particles_pos: Union[torch.tensor, None] = None, particles_vel: Union[torch.tensor, None] = None, env_ids: Union[list[int], None] = None):
        # Sets the particles' positions and velocities to the given array. Positions and velocity set as zero by default
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        for i in env_ids:
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{i}/particles")))

            # Resets particles if given
            if particles_pos is not None:
                particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(particles_pos.cpu().numpy()))
            else:
                particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.initial_particles_pos.cpu().numpy()))

            # Resets velocities if given
            if particles_vel is not None:
                particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(particles_vel.cpu().numpy()))
            else:
                particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(self.initial_particles_vel.cpu().numpy()))

    def create_pbd_material(self, mat_name: str, color_rgb: Gf.Vec3f = Gf.Vec3f(0.2, 0.2, 0.8)) -> Sdf.Path:
        # create material for particles
        create_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=create_list,
            bind_selected_prims=False,
            select_new_prim=False,
        )
        target_path = "/World/Looks/" + mat_name
        if create_list[0] != target_path:
            omni.kit.commands.execute("MovePrims", paths_to_move={create_list[0]: target_path})
        shader = UsdShade.Shader.Get(self.stage, target_path + "/Shader")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color_rgb)
        return Sdf.Path(target_path)
    

    def initialize_fluid_data(self, env_0_origin: torch.Tensor = torch.tensor([0, 0, 0], device='cuda')):
        # Fill internal fluid data after the fluid is spawned if not initialized
        if self.initial_particles_pos is None:
            # Number of particles
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{0}/particles")))
            self.initial_particles_pos = torch.from_numpy(np.asarray(particles.GetPointsAttr().Get())).cuda()
            self.initial_particles_vel = torch.from_numpy(np.asarray(particles.GetVelocitiesAttr().Get())).cuda()
            self.num_particles = self.initial_particles_pos.shape[0]

            # Initial positions and velocities
            self.initial_particles_pos = self.get_particles_position([0]) - env_0_origin # Translate to the environment 0 origin
            self.initial_particles_vel = self.get_particles_velocity([0])
