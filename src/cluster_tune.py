import numpy as np
import logging
import time
import multiprocessing as mp
from amuse.lab import Particles, nbody_system, BHTree, new_king_model
from amuse.units import units
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("DE_progress")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("de_cluster.log", mode="w")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - [%(process)d] - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(console := logging.StreamHandler())
console.setLevel(logging.INFO)
console.setFormatter(formatter)

TARGET_SIGMA = 5.1
TARGET_RC = 1.6

def run_king(params):
    pid = mp.current_process().pid
    logger.info(f"Worker {pid}: Starting run_king with params={params}")
    W0, r_c = params
    converter = nbody_system.nbody_to_si(5e5 | units.MSun, r_c | units.parsec)
    particles = new_king_model(5000, W0, convert_nbody=converter)
    sim = BHTree(converter, number_of_workers=1, use_mpi=False)
    sim.parameters.epsilon_squared = (0.05 | units.parsec)**2
    sim.particles.add_particles(particles)
    sim.evolve_model(0.5 | units.Myr)  # Faster
    sigma = np.std(sim.particles.velocity.value_in(units.kms))
    pos = sim.particles.position.value_in(units.parsec)
    vel = sim.particles.velocity.value_in(units.kms)
    r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)
    sim.stop()
    return sigma, pos, None, vel, r

def fitness(params):
    sigma, _, _, _, r = run_king(params)
    r_c_sim = np.sort(r)[len(r) // 2]
    return (sigma - TARGET_SIGMA)**2 / TARGET_SIGMA**2 + (r_c_sim - TARGET_RC)**2 / TARGET_RC**2

def callback(xk, convergence):
    current_fitness = fitness(xk)
    sigma, _, _, _, r = run_king(xk)
    r_c_sim = np.sort(r)[len(r) // 2]
    logger.info(f"Iteration {callback.iteration}: W0={xk[0]:.2f}, r_c={xk[1]:.2f} pc, Sigma={sigma:.2f} km/s, r_c_sim={r_c_sim:.2f} pc, Fitness={current_fitness:.4e}")
    callback.iteration += 1

callback.iteration = 0

logger.info("Starting DE optimization for M13")
start_time = time.time()
result = differential_evolution(fitness, [(1, 10), (1.0, 2.5)], popsize=8, maxiter=30, tol=5e-5, workers=4)
logger.info(f"DE completed: Success={result.success}, W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc, Fitness={result.fun:.4e}, Time={time.time() - start_time:.2f} s")

best_sigma, best_pos, _, best_vel, best_r = run_king(result.x)
logger.info(f"Best Sigma: {best_sigma:.2f} km/s")

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.hist(best_vel.flatten(), bins=20, color='red', alpha=0.7)
plt.axvline(TARGET_SIGMA, color='green', linestyle='--', label=f'Target σ={TARGET_SIGMA}')
plt.xlabel('Velocity (km/s)')
plt.title(f'σ={best_sigma:.2f} km/s')
plt.legend()
plt.subplot(122)
bins = np.linspace(0, np.max(best_r), 20)
hist, bin_edges = np.histogram(best_r, bins=bins, density=True)
plt.plot((bin_edges[:-1] + bin_edges[1:]) / 2, hist, 'b-', label='Simulated')
plt.axvline(TARGET_RC, color='green', linestyle='--', label=f'Target r_c={TARGET_RC}')
plt.xlabel('Radius (pc)')
plt.title('Density Profile')
plt.legend()
plt.tight_layout()
plt.savefig("m13_cluster_plots.png")