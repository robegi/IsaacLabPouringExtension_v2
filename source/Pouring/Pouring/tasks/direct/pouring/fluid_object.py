from omni.physx.scripts import physicsUtils, particleUtils, utils
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
import torch
from typing import Union


class FluidObjectCfg():

    # Number of particles along the hoorizontal and vertical axes (for direct spawn)
    numParticlesX: int
    numParticlesY: int
    numParticlesZ: int
    particleSpacing: float
     
    # Fluid properties
    particle_mass : float
    density : float
    viscosity: float 

    # Environment parameters
    num_envs: int

     

class FluidObject():

    cfg: FluidObjectCfg

    def __init__(self, cfg: FluidObjectCfg, lower_pos: Gf.Vec3f):
        self.cfg = cfg
        self.lower_pos = lower_pos # Lower position for the spawn
        
        # Scene infos (Default values)
        context = omni.usd.get_context()
        self.stage = context.get_stage()
        self.default_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path("/World")).GetPrim()
        self.stage.SetDefaultPrim(self.default_prim)
        self.default_prim_path = self.stage.GetDefaultPrim().GetPath()
        self.scenePath = Sdf.Path("/physicsScene")

        # Data from config
        self.particles_num = self.cfg.numParticlesX * self.cfg.numParticlesY * self.cfg.numParticlesZ

    def spawn_fluid(self):
            # Spawns the fluid particles in the environment 0
            env_id = 0
            
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

            # Saves the particles' initial state 
            self.initial_particles_pos = self.get_particles_position(0)
            self.initial_particles_vel = self.get_particles_velocity(0)


    def get_particles_position(self, env_ids: int = -1) -> torch.Tensor:
        # Gets particles' positions in the input environment and velocities and outputs them as torch tensors
        if env_ids == -1:
            env_ids = self.cfg.num_envs

        particles_pos = torch.zeros((env_ids, self.particles_num, 3), device='cuda')

        for i in range(env_ids):
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{i}/particles")))
            particles_pos[i] = torch.from_numpy(np.asarray(particles.GetPointsAttr().Get())).cuda()

        return particles_pos
    
    def get_particles_velocity(self, env_ids: int = -1) -> torch.Tensor:
        # Gets particles' velocities in the input environment and outputs them as torch tensors
        if env_ids == -1:
            env_ids = self.cfg.num_envs

        particles_vel = torch.zeros((env_ids, self.particles_num, 3), device='cuda')

        # Cycle through all environments
        for i in range(env_ids):
            particles = UsdGeom.Points(self.stage.GetPrimAtPath(Sdf.Path(f"/World/envs/env_{i}/particles")))
            particles_vel[i] = torch.from_numpy(np.asarray(particles.GetVelocitiesAttr().Get())).cuda()

        return particles_vel

    def set_particles_position_and_velocity(self, particles_pos: Union[torch.tensor, None] = None, particles_vel: Union[torch.tensor, None] = None, env_ids: Union[list[int], None] = None):
        # Sets the particles' positions and velocities to the given array. Positions and velocity set as zero by default
        if env_ids is not None:
            
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

