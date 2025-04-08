import numpy as np
import logging
import pickle
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
import time

from pulsar_data import load_pulsar_data, interpolate_residuals

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

N_PULSARS = 20
N_TIMES = 1000
F_YR = 1 / (365.25 * 86400)  # Frequency corresponding to 1 year (in Hz)

def glitch_dm_model(params, times_i):
    glitch_amplitude, dm_amplitude, glitch_time, dm_time, dm_slope = params
    glitch = glitch_amplitude * (times_i > glitch_time).astype(float)
    dm = dm_amplitude * (times_i > dm_time).astype(float) + dm_slope * (times_i - dm_time)
    return glitch + dm

# Load the data
data_dir = "/home/wesley/data"
cache_file = "pulsar_data_cache.pkl"
times, residuals, uncertainties, positions = load_pulsar_data(
    data_dir, n_pulsars=N_PULSARS, fit_toas=False, cache_file=cache_file
)

# Debug: Print shapes
print(f"Number of pulsars: {positions.shape[0]}")
print(f"times shape: {times.shape}")
print(f"residuals shape: {residuals.shape}")
print(f"uncertainties shape: {uncertainties.shape}")
print(f"positions shape: {positions.shape}")

# Compute angular separations
angles = []
for i in range(N_PULSARS):
    for j in range(i + 1, N_PULSARS):
        ra_i, dec_i = positions[i]
        ra_j, dec_j = positions[j]
        cos_theta = np.sin(dec_i) * np.sin(dec_j) + np.cos(dec_i) * np.cos(dec_j) * np.cos(ra_i - ra_j)
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angles.append(theta)

angles = np.array(angles)
logging.info(f"Angles sample: {angles[:4]}")

# Interpolate residuals
times, residuals, uncertainties = interpolate_residuals(times, residuals, uncertainties, N_TIMES)

# Fit glitch and DM for each pulsar
fit_results = []
for i in range(N_PULSARS):
    logging.info(f"Fitting glitch and DM for pulsar {i+1}")
    print(f"Fitting glitch and DM for pulsar {i+1}")

    mask = ~np.isnan(residuals[i])

    def objective(params, times_i=times[i], residuals_i=residuals[i], mask=mask):
        model = glitch_dm_model(params, times_i)
        diff = model - residuals_i
        return diff[mask]

    x0 = [0, 0, 5 * 365.25 * 86400, 1 * 365.25 * 86400, 0]
    result = least_squares(objective, x0=x0)
    fit_results.append(result)

# Inspect the fitted parameters
for i, result in enumerate(fit_results):
    print(f"Pulsar {i+1} fit results:")
    print(f"  Glitch amplitude: {result.x[0]:.2e} s")
    print(f"  DM amplitude: {result.x[1]:.2e} s")
    print(f"  Glitch time: {result.x[2]/(365.25*86400):.2f} years")
    print(f"  DM time: {result.x[3]/(365.25*86400):.2f} years")
    print(f"  DM slope: {result.x[4]:.2e} s/s")

# Subtract the fitted model from the residuals
for i in range(N_PULSARS):
    params = fit_results[i].x
    model = glitch_dm_model(params, times[i])
    residuals[i] -= model

# Prepare residuals for GWB fitting
residuals_gw = residuals.copy()
logger = logging.getLogger()

# Step 2: Fit GWB + Earth term

# Define the Hellings-Downs curve
def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, None)  # Prevent log(0)
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5

# Generate phases for Earth term (fixed across realizations)
phases_earth = np.random.rand(N_TIMES)

def gw_model(params, times, n_realizations=100):
    A_gw, gamma, A_earth, gamma_earth = params
    freqs = np.fft.fftfreq(N_TIMES, times[0, 1] - times[0, 0])
    mask = freqs != 0

    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)

    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)

    hd_target_local = np.eye(N_PULSARS)  # Diagonal = 1.0
    k = 0
    for i in range(N_PULSARS):
        for j in range(i + 1, N_PULSARS):
            zeta = angles[k]
            hd_value = hd_curve(zeta)
            hd_target_local[i, j] = hd_value
            hd_target_local[j, i] = hd_value
            k += 1

    eigenvalues, eigenvectors = np.linalg.eigh(hd_target_local)
    eigenvalues = np.where(eigenvalues < 0, 1e-12, eigenvalues)
    hd_target_local = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    hd_target_local = (hd_target_local + hd_target_local.T) / 2

    eigenvalues = np.linalg.eigvalsh(hd_target_local)
    logger.info(f"Eigenvalues before Cholesky: {eigenvalues[:5]}")
    print(f"Eigenvalues before Cholesky: {eigenvalues[:5]}")

    L = np.linalg.cholesky(hd_target_local)

    gw_only = np.zeros((N_PULSARS, N_TIMES))
    for _ in range(n_realizations):
        random_signals = np.random.normal(0, 1, (N_PULSARS, N_TIMES))
        gw_temp = np.dot(L, random_signals)
        gw_only += gw_temp / n_realizations
    gw_only = gw_only / np.std(gw_only) * A_gw

    hd_target_normalized = hd_target_local.copy()
    off_diag = hd_target_normalized[~np.eye(N_PULSARS, dtype=bool)]
    min_hd = np.min(off_diag)
    max_hd = np.max(off_diag)
    hd_target_normalized[~np.eye(N_PULSARS, dtype=bool)] = (off_diag - min_hd) / (max_hd - min_hd)

    earth_only = np.zeros((N_PULSARS, N_TIMES))
    model = np.zeros((N_PULSARS, N_TIMES))
    earth = np.real(np.fft.ifft(np.sqrt(power_earth) * np.exp(2j * np.pi * phases_earth)))
    for i in range(N_PULSARS):
        earth_only[i] = earth / np.std(earth) * A_earth
        model[i] = gw_only[i] + earth_only[i]

    return model, gw_only, earth_only, hd_target_normalized

eval_counter = 0

def gw_fitness(params):
    global eval_counter
    start = time.time()
    model, gw_only, earth_only, hd_target_local = gw_model(params, times)
    chi2 = np.nansum((residuals_gw - model)**2 / uncertainties**2) / (N_PULSARS * N_TIMES)
    gw_centered = gw_only - np.mean(gw_only, axis=1, keepdims=True)
    gw_std = np.std(gw_only, axis=1, keepdims=True)
    corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
    np.fill_diagonal(corr, 1)
    corr_off_diag = corr[~np.eye(N_PULSARS, dtype=bool)]
    min_corr = np.min(corr_off_diag)
    max_corr = np.max(corr_off_diag)
    corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
    angles_off_diag = np.zeros(N_PULSARS * (N_PULSARS - 1) // 2)
    k = 0
    for i in range(N_PULSARS):
        for j in range(i + 1, N_PULSARS):
            angles_off_diag[k] = angles[k]
            k += 1
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

def callback(xk, convergence):
    elapsed_time = time.time() - callback.start_time
    logger.info(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    print(f"Iteration {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
          f"Convergence={convergence:.2e}, Elapsed Time={elapsed_time:.2f}s")
    callback.iter += 1
callback.iter = 0

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

best_gw, best_gw_only, best_earth, hd_target_final = gw_model(result.x, times, n_realizations=150)
best_model = best_gw.copy()

for i in range(N_PULSARS):
    white_noise = np.random.normal(0, uncertainties[i], N_TIMES)
    best_model[i] += white_noise

plt.figure(figsize=(12, 6))
for i in range(min(3, N_PULSARS)):
    plt.plot(times[i] / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Data")
    plt.plot(times[i] / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

gw_centered = best_gw_only - np.mean(best_gw_only, axis=1, keepdims=True)
gw_std = np.std(best_gw_only, axis=1, keepdims=True)
corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
np.fill_diagonal(corr, 1)
corr_off_diag = corr[~np.eye(N_PULSARS, dtype=bool)]
min_corr = np.min(corr_off_diag)
max_corr = np.max(corr_off_diag)
corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
angles_off_diag = np.zeros(N_PULSARS * (N_PULSARS - 1) // 2)
k = 0
for i in range(N_PULSARS):
    for j in range(i + 1, N_PULSARS):
        angles_off_diag[k] = angles[k]
        k += 1
hd_theoretical = hd_curve(angles_off_diag)
min_theoretical = np.min(hd_theoretical)
max_theoretical = np.max(hd_theoretical)
hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)
correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
corr[~np.eye(N_PULSARS, dtype=bool)] = corr_off_diag * correction_factor
plt.scatter(angles * 180 / np.pi, corr[~np.eye(N_PULSARS, dtype=bool)], alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
hd_values = hd_curve(zeta)
hd_values = (hd_values - np.min(hd_values)) / (np.max(hd_values) - np.min(hd_values))
plt.plot(zeta * 180 / np.pi, hd_values, "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (degrees)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()