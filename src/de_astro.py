import numpy as np
import logging
from amuse.lab import Particles, nbody_system, BHTree, new_king_model
from amuse.units import units
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    filename="de_cluster.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger()

# Target velocity dispersion (M13 from Harris Catalog)
target_sigma = 5.0  # km/s

def run_king(params):
    """Simulate a King model and return velocity dispersion"""
    W0, r_c = params
    # Create King model: N=100, total mass=1e5 Msun, core radius=r_c parsec
    converter = nbody_system.nbody_to_si(1e5 | units.MSun, r_c | units.parsec)
    particles = new_king_model(100, W0, convert_nbody=converter)
    
    # Evolve with BHTree for a short time to stabilize
    sim = BHTree(converter)
    sim.parameters.epsilon_squared = (0.01 | units.parsec)**2
    sim.particles.add_particles(particles)
    sim.evolve_model(0.1 | units.Myr)  # ~0.1 Myr to settle
    sigma = np.std(sim.particles.velocity.value_in(units.kms))  # km/s
    sim.stop()
    return sigma, particles.position.value_in(units.parsec), particles.velocity.value_in(units.kms)

def fitness(params):
    """Fitness: difference between simulated and target dispersion"""
    sigma, _, _ = run_king(params)
    return (sigma - target_sigma)**2

def callback(xk, convergence):
    """Log DE progress"""
    current_fitness = fitness(xk)
    sigma = run_king(xk)[0]
    if callback.iteration % 5 == 0:  # Log every 5 iterations
        logger.info(f"Iteration {callback.iteration}: W0={xk[0]:.2f}, r_c={xk[1]:.2f} pc, Sigma={sigma:.2f} km/s, Fitness={current_fitness:.4e}")
    callback.iteration += 1

callback.iteration = 0

# DE optimization
bounds = [(1, 10), (0.1, 2)]  # W0 (1–10), r_c (0.1–2 parsec)
logger.info("Starting DE optimization")
result = differential_evolution(
    fitness,
    bounds,
    popsize=15,
    maxiter=50,
    workers=1,  # Single worker to keep memory low
    callback=callback,
)
logger.info(f"DE completed: Success={result.success}, W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc, Fitness={result.fun:.4e}")

# Final simulation with best parameters
best_sigma, best_pos, best_vel = run_king(result.x)
logger.info(f"Best Sigma: {best_sigma:.2f} km/s")

# Visualization
plt.figure(figsize=(10, 5))

# Position scatter (x vs y)
plt.subplot(121)
plt.scatter(best_pos[:, 0], best_pos[:, 1], s=5, c='blue', alpha=0.5)
plt.xlabel('x (pc)')
plt.ylabel('y (pc)')
plt.title(f'Cluster Positions (W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc)')

# Velocity histogram
plt.subplot(122)
plt.hist(best_vel.flatten(), bins=20, color='red', alpha=0.7)
plt.axvline(target_sigma, color='green', linestyle='--', label=f'Target σ={target_sigma} km/s')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Count')
plt.title(f'Velocity Dispersion (σ={best_sigma:.2f} km/s)')
plt.legend()

plt.tight_layout()
plt.savefig("cluster_result.png")
plt.show()