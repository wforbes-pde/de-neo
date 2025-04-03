import numpy as np
import logging
from scipy.optimize import differential_evolution

logger = logging.getLogger("DE_test")
logger.setLevel(logging.INFO)
fh = logging.FileHandler("de_test.log", mode="w")
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")
fh.setFormatter(formatter)
logger.addHandler(logging.StreamHandler())

def simple_fitness(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

def callback(xk, convergence):
    logger.info(f"Iteration {callback.iteration}: x={xk}, Fitness={simple_fitness(xk):.4e}")
    callback.iteration += 1

callback.iteration = 0

logger.info("Starting simple DE test")
result = differential_evolution(simple_fitness, [(0, 5), (0, 5)], popsize=5, maxiter=10, workers=4, callback=callback)
logger.info(f"Completed: Success={result.success}, x={result.x}, Fitness={result.fun:.4e}")