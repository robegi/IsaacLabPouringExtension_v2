class FluidObjectCfg:
    def __init__(
        self,
        prim_path: str = "/World/envs/env_.*",
        sampled_shape: str = "Cylinder",
        numParticlesX: int = 5,
        numParticlesY: int = 5,
        numParticlesZ: int = 5,
        scale_x: float = 0.1,
        scale_y: float = 0.1,
        scale_z: float = 0.4,
        particle_mass: float = 0.001,
        density: float = 0.0,
        viscosity: float = 0.91,
        cohesion: float = 5.0,
        surface_tension: float = 0.02,
        friction: float = 1.0,
        damping: float = 0.9,
        particleSpacing: float = 0.005,
        num_envs: int = 1,
        isosurface: bool = False,
        hidden_particles: bool = False,
        anisotropy: bool = False,
        smoothing: bool = False,
        transparency: bool = False,
    ):
        # Prim path
        self.prim_path = prim_path

        # Sampled shape
        self.sampled_shape = sampled_shape.capitalize()

        # Number of particles along axes (direct spawn)
        self.numParticlesX = numParticlesX
        self.numParticlesY = numParticlesY
        self.numParticlesZ = numParticlesZ

        # Cylinder dimensions (sampled spawn)
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        # Fluid properties
        self.particle_mass = particle_mass
        self.density = density
        self.viscosity = viscosity
        self.cohesion = cohesion
        self.surface_tension = surface_tension
        self.friction = friction
        self.damping = damping
        self.particleSpacing = particleSpacing

        # Environment parameters
        self.num_envs = num_envs

        # Render settings
        self.isosurface = isosurface
        self.hidden_particles = hidden_particles
        self.anisotropy = anisotropy
        self.smoothing = smoothing
        self.transparency = transparency
