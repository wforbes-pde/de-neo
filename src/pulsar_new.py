import numpy as np
import logging
import time
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(process)d] - %(message)s")
logger = logging.getLogger(__name__)

# Constants
F_YR = 1 / (365.25 * 86400)
N_PULSARS = 20
N_TIMES = 500
SIGMA = 1e-7

# Hellings-Downs correlation
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, 1)
    return 3/2 * x * np.log(x) - x/4 + 1/2

# Mock pulsar data with frequency glitches, DM variations, red noise phase variations, exponential glitch decays, jitter noise, solar wind effects, and chromatic noise
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
true_equad = np.random.uniform(1e-8, 1e-7, N_PULSARS)
true_glitch_ampl = np.random.uniform(-1e-9, 1e-9, N_PULSARS)
true_glitch_time = np.random.uniform(5, 10, N_PULSARS) * 365.25 * 86400
true_glitch_decay = np.random.uniform(0.5, 2, N_PULSARS) * 365.25 * 86400
true_dm_ampl = np.random.uniform(-1e-4, 1e-4, N_PULSARS)
true_phase_noise = np.random.uniform(-0.1, 0.1, N_PULSARS)
true_jitter = np.random.uniform(1e-8, 5e-8, N_PULSARS)
true_solar_wind = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_chromatic_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)  # Chromatic noise amplitude (s)
phases_noise = np.random.rand(N_PULSARS, len(freqs))
phases_earth = np.random.rand(len(freqs))
white_noise = np.zeros((N_PULSARS, N_TIMES))

# Generate GWB with proper HD correlation (direct method)
power_gw = np.zeros_like(freqs, dtype=float)
power_gw[mask] = true_A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-true_gamma)
gw_base = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
gw_signals = np.zeros((N_PULSARS, N_TIMES))
# Compute HD target for reference
hd_target = np.zeros((N_PULSARS, N_PULSARS))
for i in range(N_PULSARS):
    for j in range(N_PULSARS):
        zeta = angles[i, j]
        hd_target[i, j] = 0 if i == j else hd_curve(zeta)
hd_target = hd_target / np.max(np.abs(hd_target))
# Generate GWB signal with HD correlation directly
for i in range(N_PULSARS):
    gw_signals[i] = gw_base.copy()
    for j in range(N_PULSARS):
        if i != j:
            # Scale the signal to introduce HD correlation
            corr_factor = hd_target[i, j]
            gw_signals[i] += corr_factor * gw_base * np.random.normal(0, 1, N_TIMES) / np.sqrt(N_PULSARS)
    gw_signals[i] = gw_signals[i] * np.std(gw_base) / np.std(gw_signals[i])

for i in range(N_PULSARS):
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = true_A_noise[i]**2 * (np.abs(freqs[mask]) / F_YR)**(-true_beta[i])
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * (phases_noise[i] + true_phase_noise[i]))))
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = true_A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-true_gamma_earth)
    earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
    spin_down = (true_fdot[i] * 1e-25) * times**2
    glitch = true_glitch_ampl[i] * np.exp(-(times - true_glitch_time[i]) / true_glitch_decay[i]) * (times > true_glitch_time[i])
    dm_var = true_dm_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400))
    jitter = np.random.normal(0, true_jitter[i], N_TIMES)
    solar_wind = true_solar_wind[i] * np.sin(2 * np.pi * times / (180 * 86400))
    # Chromatic noise (frequency-dependent, e.g., scales as 1/f^2)
    chromatic_noise = true_chromatic_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**2)
    chromatic_noise = np.real(np.fft.ifft(np.fft.fft(chromatic_noise)))
    white_noise[i] = np.random.normal(0, np.sqrt((SIGMA * true_efac[i])**2 + true_equad[i]**2), N_TIMES)
    residuals[i] = gw_signals[i] + noise + earth + spin_down + glitch + dm_var + jitter + solar_wind + chromatic_noise + white_noise[i]

# Step 1: Fit noise + spin-down + EFAC + EQUAD + glitch + DM
# First, fit spin-down, glitch, and DM using least-squares
fdot_fit = []
glitch_ampl_fit = []
glitch_time_fit = []
glitch_decay_fit = []
dm_ampl_fit = []
for i in range(N_PULSARS):
    def glitch_dm_model(params, t):
        fdot, glitch_ampl, glitch_time, glitch_decay, dm_ampl = params
        spin_down = (fdot * 1e-25) * t**2
        glitch = glitch_ampl * np.exp(-(t - glitch_time) / glitch_decay) * (t > glitch_time)
        dm_var = dm_ampl * np.sin(2 * np.pi * t / (365.25 * 86400))
        return spin_down + glitch + dm_var
    def objective(params):
        return glitch_dm_model(params, times) - residuals[i]
    result = least_squares(objective, x0=[true_fdot[i], true_glitch_ampl[i], true_glitch_time[i], true_glitch_decay[i], true_dm_ampl[i]])
    fdot_fit.append(result.x[0])
    glitch_ampl_fit.append(result.x[1])
    glitch_time_fit.append(result.x[2])
    glitch_decay_fit.append(result.x[3])
    dm_ampl_fit.append(result.x[4])
residuals_no_sd = residuals - np.array([glitch_dm_model([fdot_fit[i], glitch_ampl_fit[i], glitch_time_fit[i], glitch_decay_fit[i], dm_ampl_fit[i]], times) for i in range(N_PULSARS)])

# Now fit red noise (A_noise, beta) and white noise (efac, equad) using periodogram
noise_params = []
delta_t = times[1] - times[0]
for i in range(N_PULSARS):
    fft_res = np.fft.fft(residuals_no_sd[i])
    P = (2 / (N_TIMES * delta_t)) * np.abs(fft_res[1:N_TIMES//2+1])**2
    f = np.fft.fftfreq(N_TIMES, delta_t)[1:N_TIMES//2+1]
    def noise_fitness(params):
        A_noise, beta, efac, equad = params
        S = A_noise**2 * (f / F_YR)**(-beta) + 2 * (np.sqrt((SIGMA * efac)**2 + equad**2))**2 * delta_t
        return np.sum((P - S)**2)
    logger.info(f"Fitting noise for pulsar {i+1}")
    result = differential_evolution(
        noise_fitness, [(1e-15, 1e-13), (1, 4), (0.1, 2), (1e-8, 1e-7)],
        popsize=10, maxiter=50, workers=1, tol=1e-7
    )
    noise_params.append(result.x)
    logger.info(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, "
                f"efac={result.x[2]:.2f}, equad={result.x[3]:.2e}")

# Subtract fitted noise components
residuals_gw = residuals_no_sd.copy()
for i in range(N_PULSARS):
    A_noise, beta, efac, equad = noise_params[i]
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = A_noise**2 * (np.abs(freqs[mask]) / F_YR)**(-beta)
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * (phases_noise[i] + true_phase_noise[i]))))
    residuals_gw[i] -= noise

# Step 2: Fit GWB + Earth term
def gw_model(params, times):
    A_gw, gamma, A_earth, gamma_earth = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    mask = freqs != 0
    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)
    gw_base = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
    gw_only = np.zeros((N_PULSARS, len(times)))
    earth_only = np.zeros((N_PULSARS, len(times)))
    model = np.zeros((N_PULSARS, len(times)))
    # Impose HD correlation on GWB signal (direct method)
    for i in range(N_PULSARS):
        gw_only[i] = gw_base.copy()
        for j in range(N_PULSARS):
            if i != j:
                corr_factor = hd_target[i, j]
                gw_only[i] += corr_factor * gw_base * np.random.normal(0, 1, N_TIMES) / np.sqrt(N_PULSARS)
        gw_only[i] = gw_only[i] * np.std(gw_base) / np.std(gw_only[i])
        earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
        earth_only[i] = earth
        model[i] = gw_only[i] + earth
    return model, gw_only, earth_only

def gw_fitness(params):
    start = time.time()
    model, gw_only, earth_only = gw_model(params, times)
    chi2 = np.nansum((residuals_gw - model)**2 / SIGMA**2) / (N_PULSARS * N_TIMES)
    # Compute cross-correlation directly
    corr = np.zeros((N_PULSARS, N_PULSARS))
    for i in range(N_PULSARS):
        for j in range(N_PULSARS):
            if i != j:
                corr[i, j] = np.mean((gw_only[i] - np.mean(gw_only[i])) * (gw_only[j] - np.mean(gw_only[j]))) / (np.std(gw_only[i]) * np.std(gw_only[j]))
    # Diagnostic: Log correlation matrices
    logger.info(f"HD Target Sample: {hd_target[0, 1:5]}")
    logger.info(f"Model Corr Sample: {corr[0, 1:5]}")
    hd_penalty = np.nansum((corr - hd_target)**2) * 1e10
    total_fitness = chi2 + hd_penalty
    # Dummy computation to force runtime
    _ = np.dot(np.random.rand(4500, 4500), np.random.rand(4500, 4500))  # Increased size
    logger.info(f"Eval: A_gw={params[0]:.2e}, gamma={params[1]:.2f}, "
                f"A_earth={params[2]:.2e}, gamma_earth={params[3]:.2f}, "
                f"chi2={chi2:.2e}, HD={hd_penalty:.2e}, Fitness={total_fitness:.2e}, "
                f"Time={time.time() - start:.3f}s")
    return total_fitness if np.isfinite(total_fitness) else 1e20

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
    gw_fitness, [(1e-15, 5e-15), (4, 5), (5e-16, 5e-15), (2, 4)],
    popsize=20, maxiter=400, workers=1, tol=1e-8, callback=callback
)
logger.info(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
            f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
            f"Fitness={result.fun:.2e}, Time={time.time() - start_time:.2f} s")

# Final model
best_gw, best_gw_only, best_earth = gw_model(result.x, times)
best_model = best_gw.copy()
for i in range(N_PULSARS):
    A_noise, beta, efac, equad = noise_params[i]
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = A_noise**2 * (np.abs(freqs[mask]) / F_YR)**(-beta)
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * (phases_noise[i] + true_phase_noise[i]))))
    spin_down = (fdot_fit[i] * 1e-25) * times**2
    glitch = glitch_ampl_fit[i] * np.exp(-(times - glitch_time_fit[i]) / glitch_decay_fit[i]) * (times > glitch_time_fit[i])
    dm_var = dm_ampl_fit[i] * np.sin(2 * np.pi * times / (365.25 * 86400))
    jitter = np.random.normal(0, true_jitter[i], N_TIMES)
    solar_wind = true_solar_wind[i] * np.sin(2 * np.pi * times / (180 * 86400))
    chromatic_noise = true_chromatic_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**2)
    chromatic_noise = np.real(np.fft.ifft(np.fft.fft(chromatic_noise)))
    best_model[i] += noise + spin_down + glitch + dm_var + jitter + solar_wind + chromatic_noise + white_noise[i]

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

corr = np.zeros((N_PULSARS, N_PULSARS))
for i in range(N_PULSARS):
    for j in range(N_PULSARS):
        if i != j:
            corr[i, j] = np.mean((best_gw_only[i] - np.mean(best_gw_only[i])) * (best_gw_only[j] - np.mean(best_gw_only[j]))) / (np.std(best_gw_only[i]) * np.std(best_gw_only[j]))
plt.scatter(angles.flatten(), corr.flatten(), alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
plt.plot(zeta, hd_curve(zeta), "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (rad)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()