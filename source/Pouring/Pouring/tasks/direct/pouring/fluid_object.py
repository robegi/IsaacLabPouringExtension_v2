from omni.physx.scripts import physicsUtils, particleUtils, utils
from isaaclab.sim.utils import get_current_stage_id
from pxr import Usd, UsdLux, UsdGeom, Sdf, Gf, Vt, UsdPhysics, PhysxSchema
import omni.physx.bindings._physx as physx_settings_bindings
import omni.timeline
import numpy as np
import omni.kit.commands
import torch

from isaacsim.core.api import SimulationContext

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
        
    
    def spawn_fluid_direct(self, env_index: int = 0):
            
            # Particle System
            self.particleSystemPath = Sdf.Path(f"/World/envs/env_{env_index}").AppendChild("particleSystem")

            # Particle points
            self.particlesPath = Sdf.Path(f"/World/envs/env_{env_index}/particles")

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

            # Set particle system properties
            material_path = omni.usd.get_stage_next_free_path(self.stage, "/pbdParticleMaterial", True)
            particleUtils.add_pbd_particle_material(self.stage, material_path)
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.stage.GetPrimAtPath(self.particleSystemPath).GetPath(),
                material_path=material_path,
                strength=None,
            )

            particle_system.CreateMaxVelocityAttr().Set(200)

            # Create Grid
            gridSpacing = self.cfg.particleSpacing + 0.001
            lower = self.lower_pos + Gf.Vec3f(-gridSpacing*self.cfg.numParticlesX/2, -gridSpacing*self.cfg.numParticlesY/2, 0)  # Translate lower corner
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

            # Get the particles' initial positions and velocities
            self.initial_particles_pos, self.initial_particles_vel = self.get_particles_position(env_index)


    def get_particles_position(self, env_id: int)->tuple[np.array, np.array]:
        # Gets particles' positions in the input environment and velocities and outputs them as arrays

        particles = UsdGeom.Points(self.stage.GetPrimAtPath(self.default_prim_path.AppendPath(f"envs/env_{env_id}/particles")))
        particles_pos = np.array(particles.GetPointsAttr().Get())
        particles_vel = np.array(particles.GetVelocitiesAttr().Get())

        return particles_pos, particles_vel

    def set_particles_position(self, particles_pos: np.array, particles_vel: np.array, env_id:int):
        # Sets the particles' position and velocities to the given arrays
        particles = UsdGeom.Points(self.stage.GetPrimAtPath(self.default_prim_path.AppendPath("envs/env_%d/particles" % env_id)))
        particles.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(particles_pos))
        particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(particles_vel))