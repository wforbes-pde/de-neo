import numpy as np
import logging
import time
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(process)d] - %(message)s")
logger = logging.getLogger(__name__)

# Constants
F_YR = 1 / (365.25 * 86400)  # 1 yr^-1 in Hz
N_PULSARS = 10
N_TIMES = 100
SIGMA = 1e-7  # Noise (100 ns)

# Mock pulsar data
np.random.seed(42)
times = np.linspace(0, 15 * 365.25 * 86400, N_TIMES)  # 15 yr
angles = np.random.uniform(0, np.pi, (N_PULSARS, N_PULSARS))  # Angular separations
residuals = np.random.normal(0, SIGMA, (N_PULSARS, N_TIMES))  # Noise
true_A = 2.4e-15
true_gamma = 13/3
for i in range(N_PULSARS):
    freqs = np.fft.fftfreq(N_TIMES, times[1] - times[0])
    power = true_A**2 / (12 * np.pi**2) * (np.abs(freqs) / F_YR)**(-true_gamma)
    gw_signal = np.fft.ifft(np.sqrt(power) * np.exp(2j * np.pi * np.random.rand(len(freqs))))
    residuals[i] += np.real(gw_signal)  # Add GWB

# Hellings-Downs correlation
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    return 3/2 * x * np.log(x) - x/4 + 1/2

# GWB model
def gw_residuals(params, times):
    A_gw, gamma = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    power = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs) / F_YR)**(-gamma)
    gw = np.zeros((N_PULSARS, len(times)))
    for i in range(N_PULSARS):
        phase = np.random.rand(len(freqs))  # Random phase per pulsar
        gw[i] = np.real(np.fft.ifft(np.sqrt(power) * np.exp(2j * np.pi * phase)))
    return gw

# Fitness function
def fitness(params):
    A_gw, gamma = params
    model = gw_residuals(params, times)
    chi2 = np.sum((residuals - model)**2 / SIGMA**2)
    # HD correlation check
    corr = np.corrcoef(model)
    hd_target = hd_curve(angles)
    hd_penalty = np.sum((corr - hd_target)**2) * 1e6  # Weight HD fit
    return chi2 + hd_penalty

# DE optimization
logger.info("Starting DE for GWB")
start_time = time.time()
result = differential_evolution(
    fitness, [(1e-16, 1e-14), (2, 5)], popsize=8, maxiter=10, workers=4
)
logger.info(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
            f"Fitness={result.fun:.2e}, Time={time.time() - start_time:.2f} s")

# Plot results
best_model = gw_residuals(result.x, times)
plt.figure(figsize=(10, 5))
for i in range(min(3, N_PULSARS)):
    plt.plot(times / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Data")
    plt.plot(times / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

# Correlation plot
corr = np.corrcoef(best_model)
plt.scatter(angles.flatten(), corr.flatten(), alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
plt.plot(zeta, hd_curve(zeta), "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (rad)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")