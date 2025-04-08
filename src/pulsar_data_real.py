from pulsar_data import load_pulsar_data, compute_angular_separations, interpolate_residuals

import numpy as np
import logging
import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Setup logging to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(process)d] - %(message)s")
logger = logging.getLogger(__name__)

# Constants
F_YR = 1 / (365.25 * 86400)
N_PULSARS = 20
N_TIMES = 500

# Hellings-Downs correlation
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, 1)
    result = 0.5 + 0.75 * x * np.log(x) - 0.25 * x
    return np.clip(result, 0, np.inf)

# Load real pulsar data (replace with actual file paths)
file_list = [f"pulsar_{i+1}_residuals.txt" for i in range(N_PULSARS)]  # Placeholder
times_list, residuals, uncertainties, positions = load_pulsar_data(file_list, N_PULSARS)
angles = compute_angular_separations(positions)
logger.info(f"Angles sample: {angles[0, 1:5]}")
print(f"Angles sample: {angles[0, 1:5]}")

# Interpolate residuals onto a regular grid
times, residuals, uncertainties = interpolate_residuals(times_list, residuals, uncertainties, N_TIMES)

# Step 1: Fit noise + spin-down + EFAC + EQUAD + glitch + DM
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
        mask = ~np.isnan(residuals[i])
        return (glitch_dm_model(params, times) - residuals[i])[mask]
    logger.info(f"Fitting glitch and DM for pulsar {i+1}")
    print(f"Fitting glitch and DM for pulsar {i+1}")
    result = least_squares(objective, x0=[0, 0, 5 * 365.25 * 86400, 1 * 365.25 * 86400, 0])
    fdot_fit.append(result.x[0])
    glitch_ampl_fit.append(result.x[1])
    glitch_time_fit.append(result.x[2])
    glitch_decay_fit.append(result.x[3])
    dm_ampl_fit.append(result.x[4])
residuals_no_sd = residuals - np.array([glitch_dm_model([fdot_fit[i], glitch_ampl_fit[i], glitch_time_fit[i], glitch_decay_fit[i], dm_ampl_fit[i]], times) for i in range(N_PULSARS)])

# Fit red noise and white noise (simplified for now)
noise_params = []
for i in range(N_PULSARS):
    def noise_fitness(params):
        A_noise, beta, efac = params
        # Placeholder: Real noise modeling would use a Bayesian approach (e.g., enterprise)
        return np.nansum((residuals_no_sd[i] - np.mean(residuals_no_sd[i]))**2)  # Simplified
    logger.info(f"Fitting noise for pulsar {i+1}")
    print(f"Fitting noise for pulsar {i+1}")
    # Using SciPy's DE for now; replace with your custom DE
    result = differential_evolution(
        noise_fitness, [(1e-15, 1e-13), (1, 4), (0.1, 2)],
        popsize=10, maxiter=50, workers=1, tol=1e-7
    )
    noise_params.append(result.x)
    logger.info(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, efac={result.x[2]:.2f}")
    print(f"Pulsar {i+1} noise: A_noise={result.x[0]:.2e}, beta={result.x[1]:.2f}, efac={result.x[2]:.2f}")

# Subtract fitted noise components (simplified)
residuals_gw = residuals_no_sd.copy()

# Step 2: Fit GWB + Earth term
def gw_model(params, times, n_realizations=100):
    A_gw, gamma, A_earth, gamma_earth = params
    freqs = np.fft.fftfreq(len(times), times[1] - times[0])
    mask = freqs != 0
    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)
    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)
    gw_base = np.real(np.fft.ifft(np.sqrt(power_gw) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
    
    # Compute HD target matrix for Cholesky decomposition
    hd_target_local = np.zeros((N_PULSARS, N_PULSARS))
    for i in range(N_PULSARS):
        for j in range(N_PULSARS):
            zeta = angles[i, j]
            hd_target_local[i, j] = 1.0 if i == j else hd_curve(zeta)
    
    # Ensure positive definiteness
    eigenvalues, eigenvectors = np.linalg.eigh(hd_target_local)
    eigenvalues = np.where(eigenvalues < 0, 1e-12, eigenvalues)
    hd_target_local = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    hd_target_local = (hd_target_local + hd_target_local.T) / 2
    
    # Debug: Check eigenvalues
    eigenvalues = np.linalg.eigvalsh(hd_target_local)
    logger.info(f"Eigenvalues before Cholesky: {eigenvalues}")
    print(f"Eigenvalues before Cholesky: {eigenvalues}")
    
    # Cholesky decomposition
    L = np.linalg.cholesky(hd_target_local)
    
    # Generate correlated GW signals
    gw_only = np.zeros((N_PULSARS, len(times)))
    for _ in range(n_realizations):
        random_signals = np.random.normal(0, 1, (N_PULSARS, len(times)))
        gw_temp = np.dot(L, random_signals)
        gw_only += gw_temp / n_realizations
    gw_only = gw_only / np.std(gw_only) * A_gw
    
    # Compute HD target matrix for fitness (with normalization)
    hd_target_normalized = hd_target_local.copy()
    off_diag = hd_target_normalized[~np.eye(N_PULSARS, dtype=bool)]
    min_hd = np.min(off_diag)
    max_hd = np.max(off_diag)
    hd_target_normalized[~np.eye(N_PULSARS, dtype=bool)] = (off_diag - min_hd) / (max_hd - min_hd)
    
    earth_only = np.zeros((N_PULSARS, len(times)))
    model = np.zeros((N_PULSARS, len(times)))
    for i in range(N_PULSARS):
        earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * np.random.rand(len(freqs)))))
        earth_only[i] = earth / np.std(earth) * A_earth
        model[i] = gw_only[i] + earth_only[i]
    return model, gw_only, earth_only, hd_target_normalized

# Counter for logging frequency
eval_counter = 0

def gw_fitness(params):
    global eval_counter
    start = time.time()
    model, gw_only, earth_only, hd_target_local = gw_model(params, times)
    # Chi-squared with real uncertainties
    weights = ~np.isnan(residuals_gw)  # 1 where data exists, 0 where NaN
    chi2 = np.nansum(weights * ((residuals_gw - model) / uncertainties)**2) / np.sum(weights)
    gw_centered = gw_only - np.mean(gw_only, axis=1, keepdims=True)
    gw_std = np.std(gw_only, axis=1, keepdims=True)
    corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
    np.fill_diagonal(corr, 1)
    # Normalize model correlations to range [0, 1] and apply shape correction
    corr_off_diag = corr[~np.eye(N_PULSARS, dtype=bool)]
    min_corr = np.min(corr_off_diag)
    max_corr = np.max(corr_off_diag)
    corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
    angles_off_diag = angles[~np.eye(N_PULSARS, dtype=bool)]
    hd_theoretical = hd_curve(angles_off_diag)
    min_theoretical = np.min(hd_theoretical)
    max_theoretical = np.max(hd_theoretical)
    hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)
    correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
    corr[~np.eye(N_PULSARS, dtype=bool)] = corr_off_diag * correction_factor
    hd_penalty = np.nansum((corr - hd_target_local)**2) * 5e6
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

# Callback
def callback(xk, convergence):
    elapsed_time = time.time() - callback.start_time
    logger.info(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    print(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
          f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    callback.iter += 1
callback.iter = 0

# DE optimization for GWB (replace with your custom DE)
logger.info("Starting DE for GWB")
print("Starting DE for GWB")
callback.start_time = time.time()
# Placeholder for your custom DE
result = custom_differential_evolution(
    gw_fitness, [(1e-15, 5e-15), (4, 5), (5e-16, 5e-15), (2, 4)],
    popsize=50, maxiter=600, tol=1e-11, callback=callback
)
logger.info(f"DE completed: A_gw={result['x'][0]:.2e}, gamma={result['x'][1]:.2f}, "
            f"A_earth={result['x'][2]:.2e}, gamma_earth={result['x'][3]:.2f}, "
            f"Fitness={result['fun']:.2e}, Time={time.time() - callback.start_time:.2f} s")
print(f"DE completed: A_gw={result['x'][0]:.2e}, gamma={result['x'][1]:.2f}, "
      f"A_earth={result['x'][2]:.2e}, gamma_earth={result['x'][3]:.2f}, "
      f"Fitness={result['fun']:.2e}, Time={time.time() - callback.start_time:.2f} s")

# Final model
best_gw, best_gw_only, best_earth, hd_target_final = gw_model(result['x'], times, n_realizations=150)
best_model = best_gw.copy()

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

# HD correlation plot
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
min_theoretical = np.min(hd_theoretical)
max_theoretical = np.max(hd_theoretical)
hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)
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