import numpy as np
import logging
from amuse.lab import Particles, nbody_system, BHTree, new_king_model
from amuse.units import units
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress AMUSE/BHTree logs
logging.getLogger("amuse").setLevel(logging.WARNING)
logging.getLogger("code").setLevel(logging.WARNING)

# Custom DE logging
logger = logging.getLogger("DE_progress")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("de_cluster.log", mode="w")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)

def read_m13_data(filename="mwgc.dat"):
    with open(filename, 'r') as f:
        lines = f.readlines()
    part_iii_start = 0
    blank_count = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            blank_count += 1
        if blank_count == 2:
            part_iii_start = i + 1
            break
    for line in lines[part_iii_start:]:
        if "NGC 6205" in line:
            fields = line.split()
            sigma = float(fields[3])
            return sigma
    raise ValueError("M13 (NGC 6205) not found in mwgc.dat")

# Load M13 target
target_sigma = read_m13_data("mwgc.dat")
logger.info(f"Loaded M13 target velocity dispersion: {target_sigma} km/s")

def run_king(params):
    W0, r_c = params
    converter = nbody_system.nbody_to_si(1e5 | units.MSun, r_c | units.parsec)
    particles = new_king_model(100, W0, convert_nbody=converter)
    initial_pos = particles.position.value_in(units.parsec)  # Capture initial state
    sim = BHTree(converter, number_of_workers=1)
    sim.parameters.epsilon_squared = (0.01 | units.parsec)**2
    sim.particles.add_particles(particles)
    sim.evolve_model(0.1 | units.Myr)
    sigma = np.std(sim.particles.velocity.value_in(units.kms))
    pos = sim.particles.position.value_in(units.parsec)
    vel = sim.particles.velocity.value_in(units.kms)
    sim.stop()
    return sigma, initial_pos, pos, vel

def fitness(params):
    sigma, _, _, _ = run_king(params)
    return (sigma - target_sigma)**2

def callback(xk, convergence):
    current_fitness = fitness(xk)
    sigma = run_king(xk)[0]
    if callback.iteration % 5 == 0:
        logger.info(f"Iteration {callback.iteration}: W0={xk[0]:.2f}, r_c={xk[1]:.2f} pc, Sigma={sigma:.2f} km/s, Fitness={current_fitness:.4e}")
    callback.iteration += 1

callback.iteration = 0

# DE optimization
bounds = [(1, 10), (0.1, 2)]
logger.info("Starting DE optimization for M13")
result = differential_evolution(fitness, bounds, popsize=15, maxiter=50, workers=1, callback=callback)
logger.info(f"DE completed: Success={result.success}, W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc, Fitness={result.fun:.4e}")

# Final run
best_sigma, initial_pos, best_pos, best_vel = run_king(result.x)
logger.info(f"Best Sigma: {best_sigma:.2f} km/s")

# Plots
plt.figure(figsize=(15, 10))

# 1. Initial vs Final Positions (Bar-like)
plt.subplot(221)
plt.scatter(initial_pos[:, 0], initial_pos[:, 1], s=5, c='blue', alpha=0.5, label='Initial')
plt.scatter(best_pos[:, 0], best_pos[:, 1], s=5, c='red', alpha=0.5, label='Final')
plt.xlabel('x (pc)')
plt.ylabel('y (pc)')
plt.title(f'Initial vs Final Positions (W0={result.x[0]:.2f}, r_c={result.x[1]:.2f} pc)')
plt.legend()

# 2. Radial Density Profile
r = np.sqrt(best_pos[:, 0]**2 + best_pos[:, 1]**2 + best_pos[:, 2]**2)
bins = np.linspace(0, np.max(r), 20)
hist, bin_edges = np.histogram(r, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.subplot(222)
plt.plot(bin_centers, hist, 'b-', label='Simulated')
plt.xlabel('Radius (pc)')
plt.ylabel('Density (normalized)')
plt.title('Radial Density Profile')
plt.legend()

# 3. Velocity Histogram
plt.subplot(223)
plt.hist(best_vel.flatten(), bins=20, color='red', alpha=0.7)
plt.axvline(target_sigma, color='green', linestyle='--', label=f'Target σ={target_sigma} km/s')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Count')
plt.title(f'Velocity Dispersion (σ={best_sigma:.2f} km/s)')
plt.legend()

# 4. Velocity Dispersion vs Radius
r_bins = np.linspace(0, np.max(r), 10)
sigma_r = []
for i in range(len(r_bins) - 1):
    mask = (r >= r_bins[i]) & (r < r_bins[i+1])
    if np.sum(mask) > 0:
        sigma_r.append(np.std(best_vel[mask].flatten()))
    else:
        sigma_r.append(0)
plt.subplot(224)
plt.plot((r_bins[:-1] + r_bins[1:]) / 2, sigma_r, 'r-', label='σ(r)')
plt.axhline(target_sigma, color='green', linestyle='--', label=f'Target σ={target_sigma} km/s')
plt.xlabel('Radius (pc)')
plt.ylabel('σ (km/s)')
plt.title('Velocity Dispersion vs Radius')
plt.legend()

plt.tight_layout()
plt.savefig("m13_cluster_plots.png")