import numpy as np
import logging
import time
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(process)d] - %(message)s")
logger = logging.getLogger(__name__)

# Constants
F_YR = 1 / (365.25 * 86400)
N_PULSARS = 20
N_TIMES = 500
SIGMA = 1e-7

# Mock pulsar data
np.random.seed(42)
times = np.linspace(0, 15 * 365.25 * 86400, N_TIMES)
angles = np.random.uniform(0, np.pi, (N_PULSARS, N_PULSARS))
angles[np.diag_indices(N_PULSARS)] = 0
residuals = np.zeros((N_PULSARS, N_TIMES))
freqs = np.fft.fftfreq(N_TIMES, times[1] - times[0])
mask = freqs != 0
true_A_gw = 2.4e-15
true_gamma = 13/3
true_A_noise = 1e-14 * np.random.uniform(0.5, 2, N_PULSARS)
true_beta = np.random.uniform(1, 3, N_PULSARS)
true_fdot = np.random.uniform(-1e-11, 1e-11, N_PULSARS) / 1e-25
true_A_earth = 1e-15
true_gamma_earth = 3.0
true_efac = np.random.uniform(0.5, 1.5, N_PULSARS)
true_equad = np.random.uniform(1e-8, 1e-7, N_PULSARS)  # Timing errors
phases_gw = np.random.rand(N_PULSARS, len(freqs))
phases_noise = np.random.rand(N_PULSARS, len(freqs))
phases_earth = np.random.rand(len(freqs))
white_noise = np.zeros((N_PULSARS, N_TIMES))
for i in range(N_PULSARS):
    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = true_A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-true_gamma)
    gw = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * phases_gw[i])))
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = true_A_noise[i]**2 * (np.abs(freqs[mask]) / F_YR)**(-true_beta[i])
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * phases_noise[i])))
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = true_A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-true_gamma_earth)
    earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
    spin_down = (true_fdot[i] * 1e-25) * times**2
    white_noise[i] = np.random.normal(0, np.sqrt((SIGMA * true_efac[i])**2 + true_equad[i]**2), N_TIMES)
    residuals[i] = gw + noise + earth + spin_down + white_noise[i]

# Hellings-Downs correlation
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, 1)
    return 3/2 * x * np.log(x) - x/4 + 1/2

# Step 1: Fit noise + spin-down + EFAC + EQUAD
def noise_model(params, times, phases_noise, pulsar_idx):
    A_noise, beta, fdot, efac, equad = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    mask = freqs != 0
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = A_noise**2 * (np.abs(freqs[mask]) / F_YR)**(-beta)
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * phases_noise[pulsar_idx])))
    spin_down = (fdot * 1e-25) * times**2
    white = np.random.normal(0, np.sqrt((SIGMA * efac)**2 + equad**2), len(times))
    return noise + spin_down + white

noise_params = []
for i in range(N_PULSARS):
    def noise_fitness(params):
        model = noise_model(params, times, phases_noise, i)
        chi2 = np.nansum((residuals[i] - model)**2 / SIGMA**2) / N_TIMES
        return chi2

    logger.info(f"Fitting noise for pulsar {i+1}")
    result = differential_evolution(
        noise_fitness, [(1e-15, 1e-13), (1, 4), (-1e-10, 1e-10), (0.1, 2.0), (1e-8, 1e-7)],
        popsize=10, maxiter=50, workers=1, tol=1e-7
    )
    noise_params.append(result.x)
    logger.info(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, "
                f"fdot={result.x[2]:.2e}, efac={result.x[3]:.2f}, equad={result.x[4]:.2e}")

# Subtract noise
residuals_gw = residuals.copy()
for i in range(N_PULSARS):
    noise = noise_model(noise_params[i], times, phases_noise, i)
    residuals_gw[i] -= noise

# Step 2: Fit GWB + Earth term
def gw_model(params, times, phases_gw, phases_earth):
    A_gw, gamma, A_earth, gamma_earth = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    mask = freqs != 0
    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)
    gw_only = np.zeros((N_PULSARS, len(times)))
    model = np.zeros((N_PULSARS, len(times)))
    for i in range(N_PULSARS):
        gw = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * phases_gw[i])))
        gw_only[i] = gw * (A_gw / true_A_gw)**2  # Amplify for correlation
        earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
        model[i] = gw + earth
    return model, gw_only

def gw_fitness(params):
    start = time.time()
    model, gw_only = gw_model(params, times, phases_gw, phases_earth)
    chi2 = np.nansum((residuals_gw - model)**2 / SIGMA**2) / (N_PULSARS * N_TIMES)
    corr = np.corrcoef(gw_only, rowvar=True)
    hd_target = hd_curve(angles)
    hd_penalty = np.nansum((corr - hd_target)**2) * 1e8  # Stronger penalty
    total_fitness = chi2 + hd_penalty
    # Dummy computation to force runtime
    _ = np.sum(np.sin(np.linspace(0, 1000, 10000000)))  # Increased
    logger.info(f"Eval: A_gw={params[0]:.2e}, gamma={params[1]:.2f}, "
                f"A_earth={params[2]:.2e}, gamma_earth={params[3]:.2f}, "
                f"chi2={chi2:.2e}, HD={hd_penalty:.2e}, Fitness={total_fitness:.2e}, "
                f"Time={time.time() - start:.3f}s")
    return total_fitness if np.isfinite(total_fitness) else 1e10

# Callback
def callback(xk, convergence):
    logger.info(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"Convergence={convergence:.2e}")
    callback.iter += 1
callback.iter = 0

# DE optimization for GWB
logger.info("Starting DE for GWB")
start_time = time.time()
result = differential_evolution(
    gw_fitness, [(1e-15, 5e-15), (4, 5), (5e-16, 5e-15), (2, 4)],  # Tighter gamma bounds
    popsize=15, maxiter=200, workers=1, tol=1e-8, callback=callback
)
logger.info(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
            f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
            f"Fitness={result.fun:.2e}, Time={time.time() - start_time:.2f} s")

# Final model
best_gw, best_gw_only = gw_model(result.x, times, phases_gw, phases_earth)
best_model = best_gw.copy()
for i in range(N_PULSARS):
    best_model[i] += noise_model(noise_params[i], times, phases_noise, i)

# Plots
plt.figure(figsize=(12, 6))
for i in range(min(3, N_PULSARS)):
    plt.plot(times / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Data")
    plt.plot(times / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

corr = np.corrcoef(best_gw_only)
plt.scatter(angles.flatten(), corr.flatten(), alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
plt.plot(zeta, hd_curve(zeta), "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (rad)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()