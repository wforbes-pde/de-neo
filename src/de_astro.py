import numpy as np
import logging
from amuse.lab import Particles, nbody_system, ph4
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    filename="de_progress.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)
logger = logging.getLogger()

# Load Gaia target (subsampled)
target = np.load("target_bar_gaia.npy")[:1000]  # N = 1000

def run_amuse(initial_conditions):
    N = len(initial_conditions) // 6
    particles = Particles(N)
    ic = initial_conditions.reshape(N, 6)
    particles.position = ic[:, :3] | nbody_system.length
    particles.velocity = ic[:, 3:] | nbody_system.speed
    particles.mass = (1.0 / N) | nbody_system.mass

    sim = ph4(number_of_workers=2)  # Limit to 2 cores
    sim.parameters.epsilon_squared = (0.01 | nbody_system.length)**2
    sim.particles.add_particles(particles)
    sim.evolve_model(1.0 | nbody_system.time)
    final_pos = sim.particles.position.value_in(nbody_system.length)
    final_vel = sim.particles.velocity.value_in(nbody_system.speed)
    sim.stop()
    return np.column_stack((final_pos, final_vel))

def fitness(ic):
    final = run_amuse(ic)
    return np.mean((final - target)**2)

def callback(xk, convergence):
    """Log progress every few iterations"""
    current_fitness = fitness(xk)
    if callback.iteration % 5 == 0:  # Log every 5th iteration
        logger.info(f"Iteration {callback.iteration}: Best Fitness = {current_fitness:.4e}, Convergence = {convergence:.4f}")
    callback.iteration += 1

callback.iteration = 0  # Initialize iteration counter

# DE optimization
bounds = [(-5, 5)] * (3 * 1000) + [(-100, 100)] * (3 * 1000)
logger.info("Starting DE optimization")
result = differential_evolution(
    fitness,
    bounds,
    popsize=5,
    maxiter=10,
    workers=4,
    callback=callback,
)
logger.info(f"DE completed: Success = {result.success}, Best Fitness = {result.fun:.4e}")
best_ic = result.x
np.save("best_ic.npy", best_ic)

# Visualization
N = 1000
initial = best_ic.reshape(N, 6)
final = run_amuse(best_ic)

plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(initial[:, 0], initial[:, 1], s=1, c='blue', alpha=0.5, label='Initial')
plt.xlabel('x (nbody)')
plt.ylabel('y (nbody)')
plt.title('Initial Conditions')
plt.legend()

plt.subplot(132)
plt.scatter(final[:, 0], final[:, 1], s=1, c='red', alpha=0.5, label='Final')
plt.scatter(target[:, 0], target[:, 1], s=1, c='green', alpha=0.2, label='Target')
plt.xlabel('x (nbody)')
plt.ylabel('y (nbody)')
plt.title('Final vs Target Positions')
plt.legend()

plt.subplot(133)
plt.scatter(final[:, 3], final[:, 4], s=1, c='red', alpha=0.5, label='Final')
plt.scatter(target[:, 3], target[:, 4], s=1, c='green', alpha=0.2, label='Target')
plt.xlabel('vx (nbody)')
plt.ylabel('vy (nbody)')
plt.title('Final vs Target Velocities')
plt.legend()

plt.tight_layout()
plt.savefig("bar_formation.png")
plt.show()