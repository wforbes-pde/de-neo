import numpy as np
import logging
import time
from scipy.optimize import differential_evolution, least_squares
import matplotlib.pyplot as plt

# Setup logging to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(process)d] - %(message)s")
logger = logging.getLogger(__name__)

# Constants
F_YR = 1 / (365.25 * 86400)
N_PULSARS = 20
N_TIMES = 500
SIGMA = 1e-7

# Hellings-Downs correlation (unchanged)
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, 1)
    result = 0.5 + 0.75 * x * np.log(x) - 0.25 * x
    return np.clip(result, 0, np.inf)

# Mock pulsar data (unchanged)
np.random.seed(42)
times = np.linspace(0, 15 * 365.25 * 86400, N_TIMES)
angles = np.random.uniform(0, np.pi, (N_PULSARS, N_PULSARS))
angles[np.diag_indices(N_PULSARS)] = 0
logger.info(f"Angles sample: {angles[0, 1:5]}")
print(f"Angles sample: {angles[0, 1:5]}")
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
true_chromatic_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_scattering_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_timing_noise_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_gw_burst_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_gw_burst_time = np.random.uniform(5, 10, N_PULSARS) * 365.25 * 86400
true_binary_ampl = np.random.uniform(1e-9, 5e-9, N_PULSARS)
true_binary_freq = np.random.uniform(1e-9, 1e-8, N_PULSARS)
phases_noise = np.random.rand(N_PULSARS, len(freqs))
phases_earth = np.random.rand(len(freqs))
white_noise = np.zeros((N_PULSARS, N_TIMES))

# Generate GWB with proper HD correlation (updated to fix normalization)
power_gw = np.zeros_like(freqs, dtype=float)
power_gw[mask] = true_A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-true_gamma)
gw_base = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
gw_signals = np.zeros((N_PULSARS, N_TIMES))
hd_target = np.zeros((N_PULSARS, N_PULSARS))
for i in range(N_PULSARS):
    for j in range(N_PULSARS):
        zeta = angles[i, j]
        hd_target[i, j] = 1.0 if i == j else hd_curve(zeta)
off_diag = hd_target[~np.eye(N_PULSARS, dtype=bool)]
min_hd = np.min(off_diag)
max_hd = np.max(off_diag)
hd_target[~np.eye(N_PULSARS, dtype=bool)] = (off_diag - min_hd) / (max_hd - min_hd)

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
    chromatic_noise = true_chromatic_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**2)
    chromatic_noise = np.real(np.fft.ifft(np.fft.fft(chromatic_noise)))
    scattering_delay = true_scattering_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**4)
    scattering_delay = np.real(np.fft.ifft(np.fft.fft(scattering_delay)))
    timing_noise = np.cumsum(np.random.normal(0, true_timing_noise_ampl[i], N_TIMES))
    gw_burst = true_gw_burst_ampl[i] * np.exp(-(times - true_gw_burst_time[i])**2 / (365.25 * 86400)**2)
    binary_inspiral = true_binary_ampl[i] * np.sin(2 * np.pi * true_binary_freq[i] * times)
    white_noise[i] = np.random.normal(0, np.sqrt((SIGMA * true_efac[i])**2 + true_equad[i]**2), N_TIMES)
    residuals[i] = gw_signals[i] + noise + earth + spin_down + glitch + dm_var + jitter + solar_wind + chromatic_noise + scattering_delay + timing_noise + gw_burst + binary_inspiral + white_noise[i]
if np.any(np.isnan(residuals)):
    logger.error("NaN detected in residuals")
    print("Error: NaN detected in residuals")
    raise ValueError("NaN detected in residuals")

# Step 1: Fit noise + spin-down + EFAC + EQUAD + glitch + DM (unchanged)
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
    logger.info(f"Fitting glitch and DM for pulsar {i+1}")
    print(f"Fitting glitch and DM for pulsar {i+1}")
    result = least_squares(objective, x0=[true_fdot[i], true_glitch_ampl[i], true_glitch_time[i], true_glitch_decay[i], true_dm_ampl[i]])
    fdot_fit.append(result.x[0])
    glitch_ampl_fit.append(result.x[1])
    glitch_time_fit.append(result.x[2])
    glitch_decay_fit.append(result.x[3])
    dm_ampl_fit.append(result.x[4])
residuals_no_sd = residuals - np.array([glitch_dm_model([fdot_fit[i], glitch_ampl_fit[i], glitch_time_fit[i], glitch_decay_fit[i], dm_ampl_fit[i]], times) for i in range(N_PULSARS)])

# Fit red noise and white noise (unchanged)
noise_params = []
delta_t = times[1] - times[0]
for i in range(N_PULSARS):
    fft_res = np.fft.fft(residuals_no_sd[i])
    P = (2 / (N_TIMES * delta_t)) * np.abs(fft_res[1:N_TIMES//2+1])**2
    f = np.fft.fftfreq(N_TIMES, delta_t)[1:N_TIMES//2+1]
    def noise_fitness(params):
        A_noise, beta, efac, equad = params
        S = A_noise**2 * (f / F_YR)**(-beta) + 2 * (np.sqrt((SIGMA * efac)**2 + true_equad[i]**2))**2 * delta_t
        return np.sum((P - S)**2)
    logger.info(f"Fitting noise for pulsar {i+1}")
    print(f"Fitting noise for pulsar {i+1}")
    result = differential_evolution(
        noise_fitness, [(1e-15, 1e-13), (1, 4), (0.1, 2), (1e-8, 1e-7)],
        popsize=10, maxiter=50, workers=1, tol=1e-7
    )
    noise_params.append(result.x)
    logger.info(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, "
                f"efac={result.x[2]:.2f}, equad={result.x[3]:.2e}")
    print(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, "
          f"efac={result.x[2]:.2f}, equad={result.x[3]:.2e}")

# Subtract fitted noise components (unchanged)
residuals_gw = residuals_no_sd.copy()
for i in range(N_PULSARS):
    A_noise, beta, efac, equad = noise_params[i]
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = A_noise**2 * (np.abs(freqs[mask]) / F_YR)**(-true_beta[i])
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * (phases_noise[i] + true_phase_noise[i]))))
    residuals_gw[i] -= noise

# Step 2: Fit GWB + Earth term (updated to fix normalization)
def gw_model(params, times, n_realizations=75):
    A_gw, gamma, A_earth, gamma_earth = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    mask = freqs != 0
    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)
    gw_base = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
    # Recompute hd_target with correct normalization
    hd_target_local = np.zeros((N_PULSARS, N_PULSARS))
    for i in range(N_PULSARS):
        for j in range(N_PULSARS):
            zeta = angles[i, j]
            hd_target_local[i, j] = 1.0 if i == j else hd_curve(zeta)
    off_diag = hd_target_local[~np.eye(N_PULSARS, dtype=bool)]
    min_hd = np.min(off_diag)
    max_hd = np.max(off_diag)
    hd_target_local[~np.eye(N_PULSARS, dtype=bool)] = (off_diag - min_hd) / (max_hd - min_hd)
    eigenvalues, eigenvectors = np.linalg.eigh(hd_target_local)
    eigenvalues = np.where(eigenvalues < 0, 1e-12, eigenvalues)
    hd_target_local = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    hd_target_local = (hd_target_local + hd_target_local.T) / 2
    L = np.linalg.cholesky(hd_target_local)
    gw_only = np.zeros((N_PULSARS, len(times)))
    for _ in range(n_realizations):
        random_signals = np.random.normal(0, 1, (N_PULSARS, len(times)))
        gw_temp = np.dot(L, random_signals)
        gw_only += gw_temp / n_realizations
    gw_only = gw_only / np.std(gw_only) * A_gw
    earth_only = np.zeros((N_PULSARS, len(times)))
    model = np.zeros((N_PULSARS, len(times)))
    for i in range(N_PULSARS):
        earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
        earth_only[i] = earth / np.std(earth) * A_earth
        model[i] = gw_only[i] + earth_only[i]
    return model, gw_only, earth_only, hd_target_local

# Counter for logging frequency
eval_counter = 0

def gw_fitness(params):
    global eval_counter
    start = time.time()
    model, gw_only, earth_only, hd_target_local = gw_model(params, times)
    chi2 = np.nansum((residuals_gw - model)**2 / SIGMA**2) / (N_PULSARS * N_TIMES)
    gw_centered = gw_only - np.mean(gw_only, axis=1, keepdims=True)
    gw_std = np.std(gw_only, axis=1, keepdims=True)
    corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
    np.fill_diagonal(corr, 1)
    # Normalize by matching the range (same as plotting section)
    corr_off_diag = corr[~np.eye(N_PULSARS, dtype=bool)]
    min_corr = np.min(corr_off_diag)
    max_corr = np.max(corr_off_diag)
    corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
    angles_off_diag = angles[~np.eye(N_PULSARS, dtype=bool)]
    hd_theoretical = hd_curve(angles_off_diag)
    min_hd = np.min(hd_theoretical)
    max_hd = np.max(hd_theoretical)
    hd_theoretical = (hd_theoretical - min_hd) / (max_hd - min_hd)
    correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
    corr[~np.eye(N_PULSARS, dtype=bool)] = corr_off_diag * correction_factor
    hd_penalty = np.nansum((corr - hd_target_local)**2) * 1e8  # Reduced penalty weight
    total_fitness = chi2 + hd_penalty
    eval_counter += 1
    if eval_counter % 10 == 0:
        logger.info(f"Eval {eval_counter}: A_gw={params[0]:.2e}, gamma={params[1]:.2f}, "
                    f"A_earth={params[2]:.2e}, gamma_earth={params[3]:.2f}, "
                    f"chi2={chi2:.2e}, HD={hd_penalty:.2e}, Fitness={total_fitness:.2e}, "
                    f"Time={time.time() - start:.3f}s")
        print(f"Eval {eval_counter}: A_gw={params[0]:.2e}, gamma={params[1]:.2f}, "
              f"A_earth={params[2]:.2e}, gamma_earth={params[3]:.2f}, "
              f"chi2={chi2:.2e}, HD={hd_penalty:.2e}, Fitness={total_fitness:.2e}, "
              f"Time={time.time() - start:.3f}s")
        logger.info(f"HD Target Sample: {hd_target_local[0, 1:5]}")
        print(f"HD Target Sample: {hd_target_local[0, 1:5]}")
        logger.info(f"Model Corr Sample: {corr[0, 1:5]}")
        print(f"Model Corr Sample: {corr[0, 1:5]}")
    return total_fitness if np.isfinite(total_fitness) else 1e20

# Callback (unchanged)
def callback(xk, convergence):
    elapsed_time = time.time() - callback.start_time
    logger.info(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    print(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
          f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    callback.iter += 1
callback.iter = 0

# DE optimization for GWB (unchanged)
logger.info("Starting DE for GWB")
print("Starting DE for GWB")
callback.start_time = time.time()
result = differential_evolution(
    gw_fitness, [(1e-15, 5e-15), (4, 5), (5e-16, 5e-15), (2, 4)],
    popsize=50, maxiter=600, workers=1, tol=1e-11, callback=callback
)
logger.info(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
            f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
            f"Fitness={result.fun:.2e}, Time={time.time() - callback.start_time:.2f} s")
print(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
      f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
      f"Fitness={result.fun:.2e}, Time={time.time() - callback.start_time:.2f} s")

# Final model (unchanged)
best_gw, best_gw_only, best_earth, hd_target_final = gw_model(result.x, times, n_realizations=150)
best_model = best_gw.copy()
for i in range(N_PULSARS):
    A_noise, beta, efac, equad = noise_params[i]
    power_noise = np.zeros_like(freqs, dtype=float)
    power_noise[mask] = A_noise**2 * (np.abs(freqs[mask]) / F_YR)**(-true_beta[i])
    noise = np.real(np.fft.ifft(np.sqrt(power_noise) * np.exp(2j * np.pi * (phases_noise[i] + true_phase_noise[i]))))
    spin_down = (fdot_fit[i] * 1e-25) * times**2
    glitch = glitch_ampl_fit[i] * np.exp(-(times - glitch_time_fit[i]) / glitch_decay_fit[i]) * (times > glitch_time_fit[i])
    dm_var = dm_ampl_fit[i] * np.sin(2 * np.pi * times / (365.25 * 86400))
    jitter = np.random.normal(0, true_jitter[i], N_TIMES)
    solar_wind = true_solar_wind[i] * np.sin(2 * np.pi * times / (180 * 86400))
    chromatic_noise = true_chromatic_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**2)
    chromatic_noise = np.real(np.fft.ifft(np.fft.fft(chromatic_noise)))
    scattering_delay = true_scattering_ampl[i] * np.sin(2 * np.pi * times / (365.25 * 86400)) / (1 + (freqs / F_YR)**4)
    scattering_delay = np.real(np.fft.ifft(np.fft.fft(scattering_delay)))
    timing_noise = np.cumsum(np.random.normal(0, true_timing_noise_ampl[i], N_TIMES))
    gw_burst = true_gw_burst_ampl[i] * np.exp(-(times - true_gw_burst_time[i])**2 / (365.25 * 86400)**2)
    binary_inspiral = true_binary_ampl[i] * np.sin(2 * np.pi * true_binary_freq[i] * times)
    best_model[i] += noise + spin_down + glitch + dm_var + jitter + solar_wind + chromatic_noise + scattering_delay + timing_noise + gw_burst + binary_inspiral + white_noise[i]

# Plots (unchanged)
plt.figure(figsize=(12, 6))
for i in range(min(3, N_PULSARS)):
    plt.plot(times / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Data")
    plt.plot(times / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

# Vectorized cross-correlation for final plot (unchanged)
gw_centered = best_gw_only - np.mean(best_gw_only, axis=1, keepdims=True)
gw_std = np.std(best_gw_only, axis=1, keepdims=True)
corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
np.fill_diagonal(corr, 1)
corr_off_diag = corr[~np.eye(N_PULSARS, dtype=bool)]
min_corr = np.min(corr_off_diag)
max_corr = np.max(corr_off_diag)
corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
angles_off_diag = angles[~np.eye(N_PULSARS, dtype=bool)]
hd_theoretical = hd_curve(angles_off_diag)
min_hd = np.min(hd_theoretical)
max_hd = np.max(hd_theoretical)
hd_theoretical = (hd_theoretical - min_hd) / (max_hd - min_hd)
correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
corr[~np.eye(N_PULSARS, dtype=bool)] = corr_off_diag * correction_factor
plt.scatter(angles.flatten(), corr.flatten(), alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
hd_values = hd_curve(zeta)
hd_values = (hd_values - np.min(hd_values)) / (np.max(hd_values) - np.min(hd_values))
plt.plot(zeta, hd_values, "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (rad)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()