import carb
from omni.physx.scripts import physicsUtils, particleUtils, utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema, UsdShade
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
import torch
from typing import Union
from .fluid_object_cfg import FluidObjectCfg
import re



class FluidObject():

    cfg: FluidObjectCfg

    def __init__(self, cfg: FluidObjectCfg):
        self.cfg = cfg
                
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
        self.initialized: bool = False


    def spawn_fluid_grid(self, env_id: int = 0, pos: Gf.Vec3f | torch.Tensor | list = [0.0, 0.0, 0.0]):
            ###
            # Spawns the fluid particles in a grid (in the environment 0 by default)
            ###

            # Base position of the fluid grid
            if isinstance(pos, torch.Tensor):
                self.lower_pos = Gf.Vec3f(*pos.tolist())
            elif isinstance(pos, list):
                self.lower_pos = Gf.Vec3f(*pos)
            else:
                self.lower_pos = pos
            
            # Particle System
            particleSystemPath = self.default_prim_path.AppendChild("particleSystem")

            # Particle points path
            particlesPath = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{env_id}", self.cfg.prim_path))
            particlesPath = particlesPath.AppendChild("particles")

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
                particle_system_path=particleSystemPath,
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
                "BindMaterial", prim_path=particleSystemPath, material_path=pbd_particle_material_path
            )

            # Enables translucency to render transparent materials if required
            settings = carb.settings.get_settings()
            settings.set("/rtx/translucency/enabled", True)
            
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

            # Create Grid
            gridSpacing = self.cfg.particleSpacing + 0.001
            lower = self.lower_pos + Gf.Vec3f(-gridSpacing*self.cfg.numParticlesX/2, -gridSpacing*self.cfg.numParticlesY/2, 0) # Translate lower corner
            positions, velocities = particleUtils.create_particles_grid(
                lower, gridSpacing, self.cfg.numParticlesX, self.cfg.numParticlesY, self.cfg.numParticlesZ
            )


            widths = [self.cfg.particleSpacing] * len(positions)
            
            points = particleUtils.add_physx_particleset_points(
                stage=self.stage,
                path=particlesPath,
                positions_list=Vt.Vec3fArray(positions),
                velocities_list=Vt.Vec3fArray(velocities),
                widths_list=widths,
                particle_system_path=particleSystemPath,
                self_collision=True,
                fluid=True,
                particle_group=0,
                particle_mass=self.cfg.particle_mass,
                density=self.cfg.density,
            )

            # Render settings
            if self.cfg.anisotropy:
                # apply api and use all defaults
                PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())

            if self.cfg.smoothing:
                # apply api and use all defaults
                PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())

            if self.cfg.isosurface:
                # apply isosurface params
                isosurfaceAPI = PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(particle_system.GetPrim())
                isosurfaceAPI.CreateIsosurfaceEnabledAttr().Set(True)
                isosurfaceAPI.CreateMaxVerticesAttr().Set(1024 * 1024)
                isosurfaceAPI.CreateMaxTrianglesAttr().Set(2 * 1024 * 1024)
                isosurfaceAPI.CreateMaxSubgridsAttr().Set(1024 * 4)
                isosurfaceAPI.CreateGridSpacingAttr().Set(fluidRestOffset * 1.5)
                isosurfaceAPI.CreateSurfaceDistanceAttr().Set(fluidRestOffset * 1.6)
                isosurfaceAPI.CreateGridFilteringPassesAttr().Set("")
                isosurfaceAPI.CreateGridSmoothingRadiusAttr().Set(fluidRestOffset * 2)

                if self.cfg.anisotropy:
                    ani_api = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
                    ani_api.CreateScaleAttr().Set(5.0)
                    ani_api.CreateMinAttr().Set(1.0)  # avoids gaps in surface
                    ani_api.CreateMaxAttr().Set(2.0)

            if self.cfg.hidden_particles:
                # Hide particles
                visibility_attribute = points.GetVisibilityAttr()
                visibility_attribute.Set("invisible")


    def spawn_fluid_sampler(self, env_id: int = 0, pos: Gf.Vec3f | torch.Tensor | list = [0.0, 0.0, 0.0]):
        ###
        # Spawns fluid particles by sampling a mesh (in environment 0 by default)
        # Code reference in the isaac sim particle sampler demo
        ###

        # Base position of the fluid volume
        if isinstance(pos, torch.Tensor):
            self.lower_pos = Gf.Vec3f(*pos.tolist())
        elif isinstance(pos, list):
            self.lower_pos = Gf.Vec3f(*pos)
        else:
            self.lower_pos = pos

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
        cylinder_mesh_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{env_id}", self.cfg.prim_path))
        cylinder_mesh_path = cylinder_mesh_path.AppendChild("Cylinder")
        cylinder_resolution = (
            10  
        )
        omni.kit.commands.execute(
            "CreateMeshPrim", prim_type=self.cfg.sampled_shape, u_patches=cylinder_resolution, v_patches=cylinder_resolution, select_new_prim=False,
            prim_path = cylinder_mesh_path
        )
        cylinder_mesh = UsdGeom.Mesh.Get(self.stage, cylinder_mesh_path)
        physicsUtils.set_or_add_scale_op(cylinder_mesh, Gf.Vec3f(self.cfg.scale_x, self.cfg.scale_y, self.cfg.scale_z))
        physicsUtils.set_or_add_translate_op(cylinder_mesh, self.lower_pos) # Translate to the spawn position

        # configure target particle set:
        particle_points_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{env_id}", self.cfg.prim_path))
        particle_points_path = particle_points_path.AppendChild("particles")
        points = UsdGeom.Points.Define(self.stage, particle_points_path)
        
        # Create a pbd particle material and set it on the particle system
        pbd_particle_material_path = omni.usd.get_stage_next_free_path(self.stage, "/pbdParticleMaterial", True)
        particleUtils.add_pbd_particle_material(
            self.stage,
            pbd_particle_material_path,
            cohesion=self.cfg.cohesion,
            viscosity=self.cfg.viscosity,
            surface_tension=self.cfg.surface_tension,
            friction=self.cfg.friction,
            damping=self.cfg.damping,
        )
        physicsUtils.add_physics_material_to_prim(self.stage, particle_system.GetPrim(), pbd_particle_material_path)

        # Enable transparent material if required
        if self.cfg.transparency:
            # Set translucency to render transparent materials
            material_type = "OmniGlass"
            settings = carb.settings.get_settings()
            settings.set("/rtx/translucency/enabled", True)
        else:
            material_type = "OmniPBR"

        # Create and bind render material
        render_material_path = self.create_pbd_material(material_type)
        particleUtils.add_pbd_particle_material(self.stage, render_material_path)
        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.stage.GetPrimAtPath(particle_system_path).GetPath(),
            material_path=render_material_path,
            strength=None,
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

        # Render settings
        if self.cfg.anisotropy:
            # apply api and use all defaults
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())

        if self.cfg.smoothing:
            # apply api and use all defaults
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(particle_system.GetPrim())

        if self.cfg.isosurface:
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

            if self.cfg.anisotropy:
                ani_api = PhysxSchema.PhysxParticleAnisotropyAPI.Apply(particle_system.GetPrim())
                ani_api.CreateScaleAttr().Set(5.0)
                ani_api.CreateMinAttr().Set(1.0)  # avoids gaps in surface
                ani_api.CreateMaxAttr().Set(2.0)

        if self.cfg.hidden_particles:
            # Hide particles
            visibility_attribute = points.GetVisibilityAttr()
            visibility_attribute.Set("invisible")

    
    def get_particles_pos(self, env_ids: Union[list[int], None] = None) -> torch.Tensor:
        # Gets particles' positions in the input environment and velocities and outputs them as torch tensors
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        particles_pos = torch.zeros((len(env_ids), self.num_particles, 3), device='cuda')
        for i in env_ids:
            particle_points_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{i}", self.cfg.prim_path))
            particle_points_path = particle_points_path.AppendChild("particles")
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(particle_points_path))
            particles_pos[i] = torch.from_numpy(np.asarray(particles.GetPointsAttr().Get())).cuda().clone()

        return particles_pos
    
    def get_particles_vel(self, env_ids: Union[list[int], None] = None) -> torch.Tensor:
        # Gets particles' velocities in the input environment and outputs them as torch tensors
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        particles_vel = torch.zeros((len(env_ids), self.num_particles, 3), device='cuda')
        # Cycle through all environments
        for i in env_ids:
            particle_points_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{i}", self.cfg.prim_path))
            particle_points_path = particle_points_path.AppendChild("particles")
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(particle_points_path))
            particles_vel[i] = torch.from_numpy(np.asarray(particles.GetVelocitiesAttr().Get())).cuda().clone()

        return particles_vel

    def set_particles_pos_and_vel(self, particles_pos: Union[torch.tensor, None] = None, 
                                            particles_vel: Union[torch.tensor, None] = None, 
                                            env_ids: Union[list[int], None] = None):
        # Sets the particles' positions and velocities to the given array. Positions and velocities set as
        # default values if no argument is given

        # Unless specific ids are given, resets all envs
        if env_ids is None:
            env_ids = range(self.cfg.num_envs)

        for i in env_ids:
            particle_points_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{i}", self.cfg.prim_path))
            particle_points_path = particle_points_path.AppendChild("particles")
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(particle_points_path))
            
            # Resets particles if given
            if particles_pos is not None:
                particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(particles_pos[i].cpu().numpy()))
            else:
                particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.initial_particles_pos.cpu().numpy()))

            # Resets velocities if given
            if particles_vel is not None:
                particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(particles_vel[i].cpu().numpy()))
            else:
                particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(self.initial_particles_vel.cpu().numpy()))

    def create_pbd_material(self, mat_name: str, color_rgb: Gf.Vec3f = Gf.Vec3f(0.2, 0.2, 0.8)) -> Sdf.Path:
        # Create material for particles
        create_list = []
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name=mat_name + ".mdl",
            mtl_name=mat_name,
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
    

    def initialize_fluid_data(self, env_id: int = 0, base_env_origin: torch.Tensor = torch.tensor([0, 0, 0], device='cuda')):
        # Fill internal fluid data after the fluid is spawned if not initialized
        # The data is taken from a specific base environment
        if not self.initialized:
            # Number of particles
            particle_points_path = Sdf.Path(re.sub(r'\.\*\??|\.\+\??|\([^)]*\)', f"{env_id}", self.cfg.prim_path))
            particle_points_path = particle_points_path.AppendChild("particles")
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(particle_points_path))
            self.initial_particles_pos = torch.from_numpy(np.asarray(particles.GetPointsAttr().Get())).cuda()
            self.initial_particles_pos -= base_env_origin # Translate to the environment 0 origin
            self.initial_particles_vel = torch.from_numpy(np.asarray(particles.GetVelocitiesAttr().Get())).cuda()
            self.num_particles = self.initial_particles_pos.shape[0]

            # Flag initialization
            self.initialized = True
