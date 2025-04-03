import numpy as np
import logging
import time
from amuse.lab import Particles, nbody_system, BHTree, new_king_model
from amuse.units import units

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

TARGET_SIGMA = 5.1  # km/s
TARGET_RC = 1.6     # pc

def run_king(params, n_particles=1000, evolve_time=0.1 | units.Myr):
    start_time = time.time()
    W0, r_c = params
    logger.debug(f"Starting run_king: W0={W0}, r_c={r_c}, N={n_particles}")
    converter = nbody_system.nbody_to_si(5e5 | units.MSun, r_c | units.parsec)
    particles = new_king_model(n_particles, W0, convert_nbody=converter)
    logger.debug("Particles created")
    initial_pos = particles.position.value_in(units.parsec)
    sim = BHTree(converter, number_of_workers=1, use_mpi=False)  # Explicit no MPI
    logger.debug("BHTree initialized")
    sim.parameters.epsilon_squared = (0.05 | units.parsec)**2
    logger.debug("Epsilon set")
    sim.particles.add_particles(particles)
    logger.debug("Particles added")
    sim.evolve_model(evolve_time)
    logger.debug("Evolution completed")
    sigma = np.std(sim.particles.velocity.value_in(units.kms))
    pos = sim.particles.position.value_in(units.parsec)
    vel = sim.particles.velocity.value_in(units.kms)
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    sim.stop()
    logger.info(f"Run completed in {time.time() - start_time:.2f} s")
    return sigma, initial_pos, pos, vel, r

if __name__ == "__main__":
    logger.info("Starting test run")
    run_king([5.0, 1.6], n_particles=1000, evolve_time=0.1 | units.Myr)
    run_king([5.0, 1.6], n_particles=5000, evolve_time=0.1 | units.Myr)