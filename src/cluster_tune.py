import numpy as np
import logging
from amuse.lab import Particles, nbody_system, BHTree, new_king_model
from amuse.units import units
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename="de_cluster.log", level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

def read_m13_data(filename="mwgc.dat"):
    """Read M13 velocity dispersion from Harris Catalog mwgc.dat"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Part III starts after second blank line (around line 160–170)
    part_iii_start = 0
    blank_count = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            blank_count += 1
        if blank_count == 2:
            part_iii_start = i + 1
            break
    
    # Find M13 (NGC 6205) in Part III
    for line in lines[part_iii_start:]:
        if "NGC 6205" in line:
            # Columns: ID, V_hb, e_V_hb, sigma, e_sigma, ...
            # sigma is 4th column (index 3, fixed-width)
            fields = line.split()
            sigma = float(fields[3])  # Velocity dispersion in km/s
            return sigma
    
    raise ValueError("M13 (NGC 6205) not found in mwgc.dat")

# Load M13 target velocity dispersion
target_sigma = read_m13_data("mwgc.dat")
logger.info(f"Loaded M13 target velocity dispersion: {target_sigma} km/s")

def run_king(params):
    W0, r_c = params
    converter = nbody_system.nbody_to_si(1e5 | units.MSun, r_c | units.parsec)
    particles = new_king_model(100, W0, convert_nbody=converter)
    sim = BHTree(converter)
    sim.parameters.epsilon_squared = (0.01 | units.parsec)**2
    sim.particles.add_particles(particles)
    sim.evolve_model(0.1 | units.Myr)
    sigma = np.std(sim.particles.velocity.value_in(units.kms))
    pos = sim.particles.position.value_in(units.parsec)
    vel = sim.particles.velocity.value_in(units.kms)
    sim.stop()
    return sigma, pos, vel

def fitness(params):
    sigma, _, _ = run_king(params)
    return (sigma - target_sigma)**2

def callback(xk, convergence):
    current_fitness = fitness(xk)
    sigma = run_king(xk)[0]
    if callback.iteration % 5 == 0:
        logger.info(f"Iteration {callback.iteration}: W0={xk[0]:.2f}, r_c={xk[1]:.2f} pc, Sigma={sigma:.2f} km/s, Fitness={current_fitness:.4e}")
    callback.iteration += 1

callback.iteration = 0

# DE optimization
bounds = [(1, 10), (0.1, 2)]  # W0, r_c (parsec)
logger.info("Starting DE optimization for M13")
result = differential_evolution(fitness, bounds, popsize=15, maxiter=50, workers=1, callback=callback)
logger.info(f"DE completed: Success={result.success}, W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc, Fitness={result.fun:.4e}")

# Final run
best_sigma, best_pos, best_vel = run_king(result.x)
logger.info(f"Best Sigma: {best_sigma:.2f} km/s")

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(best_pos[:, 0], best_pos[:, 1], s=5, c='blue', alpha=0.5)
plt.xlabel('x (pc)')
plt.ylabel('y (pc)')
plt.title(f'M13 Positions (W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc)')

plt.subplot(122)
plt.hist(best_vel.flatten(), bins=20, color='red', alpha=0.7)
plt.axvline(target_sigma, color='green', linestyle='--', label=f'Target σ={target_sigma} km/s')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Count')
plt.title(f'Velocity Dispersion (σ={best_sigma:.2f} km/s)')
plt.legend()

plt.tight_layout()
plt.savefig("m13_cluster.png")
plt.show()