from amuse.lab import *
from amuse.community.bhtree import Bhtree
from amuse.units import nbody_system

converter = nbody_system.nbody_to_si(1.0 | units.MSun, 1.0 | units.AU)
particles = Particles(2)
particles.mass = [1.0, 1.0] | units.MSun
particles.position = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]] | units.AU
particles.velocity = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]] | units.kms

gravity = Bhtree(converter, channel_type="mpi")
gravity.parameters.epsilon_squared = 0.01 | units.AU**2
gravity.particles.add_particles(particles)

for t in [0.1, 0.2, 0.3] | units.yr:
    gravity.evolve_model(t)
    print(f"Time: {t}")
    print(gravity.particles)
gravity.stop()