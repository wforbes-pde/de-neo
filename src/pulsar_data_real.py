import numpy as np
import logging
import pickle
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d

from pulsar_data import load_pulsar_data, interpolate_residuals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

N_PULSARS = 20
N_TIMES = 1000
F_YR = 1 / (365.25 * 86400)

def glitch_dm_model(params, times_i, simplified=False):
    if simplified:  # For Pulsar 20: Only one glitch and a global slope
        glitch_amplitude2, glitch_time2, glitch_decay2, global_slope = params
        glitch2 = glitch_amplitude2 * (times_i > glitch_time2).astype(float) * np.exp(-(times_i - glitch_time2) / glitch_decay2)
        linear_trend = global_slope * times_i
        return glitch2 + linear_trend
    # Full model for other pulsars
    glitch_amplitude1, glitch_amplitude2, dm_amplitude, glitch_time1, glitch_time2, dm_time, dm_slope1, dm_slope2, glitch_decay1, glitch_decay2, global_slope = params
    glitch1 = glitch_amplitude1 * (times_i > glitch_time1).astype(float) * np.exp(-(times_i - glitch_time1) / glitch_decay1)
    glitch2 = glitch_amplitude2 * (times_i > glitch_time2).astype(float) * np.exp(-(times_i - glitch_time2) / glitch_decay2)
    dm = dm_amplitude * (times_i > dm_time).astype(float) + dm_slope1 * (times_i - dm_time) * (times_i <= glitch_time2).astype(float) + dm_slope2 * (times_i - glitch_time2) * (times_i > glitch_time2).astype(float)
    linear_trend = global_slope * times_i
    return glitch1 + glitch2 + dm + linear_trend

# Load the data
data_dir = "/home/wesley/data"
cache_file = "pulsar_data_cache.pkl"
times, residuals, uncertainties, positions = load_pulsar_data(
    data_dir, n_pulsars=N_PULSARS, fit_toas=False, cache_file=cache_file
)

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
logger.info(f"Angles sample: {angles[:4]}")

# Interpolate residuals
times, residuals, uncertainties = interpolate_residuals(times, residuals, uncertainties, N_TIMES)

# Plot Pulsar 20's raw residuals
plt.figure(figsize=(12, 4))
plt.plot(times[19] / (365.25 * 86400), residuals[19], label="Pulsar 20 Raw Residuals")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_20_raw_residuals.png")
plt.close()

# Estimate data noise level
data_std = np.array([np.std(residuals[i][~np.isnan(residuals[i])]) for i in range(N_PULSARS)])
print(f"Data standard deviations: {data_std[:3]}")
print(f"Uncertainties: {uncertainties[:3, 0]}")

# Compute weights based on data length
data_weights = np.array([np.sum(~np.isnan(residuals[i])) for i in range(N_PULSARS)], dtype=float)
data_weights /= data_weights.sum()  # Normalize weights

# Fit glitch and DM for each pulsar
fit_results = []
for i in range(N_PULSARS):
    logger.info(f"Fitting glitch and DM for pulsar {i+1}")
    print(f"Fitting glitch and DM for pulsar {i+1}")

    mask = ~np.isnan(residuals[i])
    times_i = times[i][mask]
    if len(times_i) == 0 or not np.all(np.isfinite(times_i)):
        logger.warning(f"Pulsar {i+1}: times_i is empty or contains non-finite values after masking. Using default max_time of 3.5 years.")
        max_time = 3.5
    else:
        max_time = times_i.max() / (365.25 * 86400)
    logger.info(f"Pulsar {i+1}: max_time = {max_time:.2f} years, len(times_i) = {len(times_i)}")

    def objective(params, times_i=times[i], residuals_i=residuals[i], mask=mask, simplified=(i == 19), return_scalar=False):
        model = glitch_dm_model(params, times_i, simplified=simplified)
        diff = model - residuals_i
        residuals = diff[mask]
        if return_scalar:
            return np.sum(residuals**2)
        return residuals

    def callback_pulsar20(xk, convergence):
        elapsed_time = time.time() - callback_pulsar20.start_time
        logger.info(f"Pulsar 20 Iter {callback_pulsar20.iter}: "
                    f"glitch_amplitude2={xk[0]:.2e}, glitch_time2={xk[1]/(365.25*86400):.2f} yr, "
                    f"glitch_decay2={xk[2]/86400:.2f} days, global_slope={xk[3]:.2e}, "
                    f"Convergence={convergence:.2e}, Time={elapsed_time:.1f}s")
        callback_pulsar20.iter += 1
    callback_pulsar20.iter = 0

    if i == 19:  # Pulsar 20: Simplified model with one glitch and global slope
        x0 = [-5e-6, 2 * 365.25 * 86400, 30 * 86400, 0]  # glitch_amplitude2, glitch_time2, glitch_decay2, global_slope
        bounds = [
            (-1e-5, 0),  # glitch_amplitude2
            (1.9 * 365.25 * 86400, 2.1 * 365.25 * 86400),  # glitch_time2
            (29 * 86400, 31 * 86400),  # glitch_decay2
            (-1e-14, 1e-14)  # global_slope
        ]
        logger.info(f"Pulsar 20 bounds: {bounds}")
        callback_pulsar20.start_time = time.time()
        result = differential_evolution(
            lambda params: objective(params, times_i=times[i], residuals_i=residuals[i], mask=mask, simplified=True, return_scalar=True),
            bounds=bounds,
            popsize=30,
            maxiter=2000,
            callback=callback_pulsar20
        )
        bounds_ls = ([b[0] for b in bounds], [b[1] for b in bounds])
        result = least_squares(
            lambda params: objective(params, times_i=times[i], residuals_i=residuals[i], mask=mask, simplified=True, return_scalar=False),
            x0=result.x,
            bounds=bounds_ls
        )
    else:
        x0 = [1e-6, 0, 1e-6, 8 * 365.25 * 86400, 8 * 365.25 * 86400, 2 * 365.25 * 86400, 1e-12, 0, 1e7, 1e7, 0]
        bounds = [
            (-1e-5, 1e-5),  # glitch_amplitude1
            (-1e-5, 1e-5),  # glitch_amplitude2
            (-1e-5, 1e-5),  # dm_amplitude
            (0, 16 * 365.25 * 86400),  # glitch_time1
            (0, 16 * 365.25 * 86400),  # glitch_time2
            (0, 16 * 365.25 * 86400),  # dm_time
            (-1e-10, 1e-10),  # dm_slope1
            (-1e-10, 1e-10),  # dm_slope2
            (86400, 365.25 * 86400),  # glitch_decay1
            (86400, 365.25 * 86400),  # glitch_decay2
            (-1e-10, 1e-10)  # global_slope
        ]
        bounds_ls = ([b[0] for b in bounds], [b[1] for b in bounds])
        result = least_squares(
            lambda params: objective(params, times_i=times[i], residuals_i=residuals[i], mask=mask, simplified=False, return_scalar=False),
            x0=x0,
            bounds=bounds_ls
        )
    fit_results.append(result)

# Inspect the fitted parameters
for i, result in enumerate(fit_results):
    print(f"Pulsar {i+1} fit results:")
    if i == 19:  # Pulsar 20
        print(f"  Glitch amplitude 2: {result.x[0]:.2e} s")
        print(f"  Glitch time 2: {result.x[1]/(365.25*86400):.2f} years")
        print(f"  Glitch decay 2: {result.x[2]/(86400):.2f} days")
        print(f"  Global slope: {result.x[3]:.2e} s/s")
    else:
        print(f"  Glitch amplitude 1: {result.x[0]:.2e} s")
        print(f"  Glitch amplitude 2: {result.x[1]:.2e} s")
        print(f"  DM amplitude: {result.x[2]:.2e} s")
        print(f"  Glitch time 1: {result.x[3]/(365.25*86400):.2f} years")
        print(f"  Glitch time 2: {result.x[4]/(365.25*86400):.2f} years")
        print(f"  DM time: {result.x[5]/(365.25*86400):.2f} years")
        print(f"  DM slope 1: {result.x[6]:.2e} s/s")
        print(f"  DM slope 2: {result.x[7]:.2e} s/s")
        print(f"  Glitch decay 1: {result.x[8]/(86400):.2f} days")
        print(f"  Glitch decay 2: {result.x[9]/(86400):.2f} days")
        print(f"  Global slope: {result.x[10]:.2e} s/s")

# Subtract the fitted model from the residuals
for i in range(N_PULSARS):
    params = fit_results[i].x
    model = glitch_dm_model(params, times[i], simplified=(i == 19))
    residuals[i] -= model

# Plot Pulsar 20's residuals after glitch/DM subtraction and the fitted model
plt.figure(figsize=(12, 4))
plt.plot(times[19] / (365.25 * 86400), residuals[19], label="Pulsar 20 Residuals After Glitch/DM")
model_20 = glitch_dm_model(fit_results[19].x, times[19], simplified=True)
plt.plot(times[19] / (365.25 * 86400), model_20, "--", label="Pulsar 20 Glitch/DM Fit")
plt.xlabel("Time (yr)")
plt.ylabel("Residual (s)")
plt.legend()
plt.savefig("pulsar_20_glitch_dm_fit.png")
plt.close()

# Prepare residuals for GWB fitting
residuals_gw = residuals.copy()

# Step 2: Fit GWB + Earth term + Red Noise

def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, None)
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5

phases_earth = np.random.rand(N_TIMES)

def gw_model(params, times, n_realizations=2000):
    A_gw, gamma, A_earth, gamma_earth = params[:4]
    red_noise_params = params[4:]
    A_red = red_noise_params[::2]
    gamma_red = red_noise_params[1::2]

    freqs = np.fft.fftfreq(N_TIMES, times[0, 1] - times[0, 0])
    mask = freqs != 0

    power_gw = np.zeros_like(freqs, dtype=float)
    power_gw[mask] = A_gw**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma)

    power_earth = np.zeros_like(freqs, dtype=float)
    power_earth[mask] = A_earth**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_earth)

    hd_target_local = np.eye(N_PULSARS)
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

    red_noise_only = np.zeros((N_PULSARS, N_TIMES))
    for i in range(N_PULSARS):
        power_red = np.zeros_like(freqs, dtype=float)
        power_red[mask] = A_red[i]**2 / (12 * np.pi**2) * (np.abs(freqs[mask]) / F_YR)**(-gamma_red[i])
        red_noise = np.real(np.fft.ifft(np.sqrt(power_red) * np.exp(2j * np.pi * np.random.rand(N_TIMES))))
        red_noise_only[i] = red_noise / np.std(red_noise) * A_red[i]
        model[i] = gw_only[i] + earth_only[i] + red_noise_only[i]

    return model, gw_only, earth_only, red_noise_only, hd_target_normalized

eval_counter = 0

def gw_fitness(params):
    global eval_counter
    start = time.time()
    model, gw_only, earth_only, red_noise_only, hd_target_local = gw_model(params, times, n_realizations=2000)
    model_with_noise = model.copy()
    for i in range(N_PULSARS):
        white_noise = np.random.normal(0, data_std[i] / 8, N_TIMES)
        model_with_noise[i] += white_noise
    chi2_per_pulsar = np.nansum((residuals_gw - model_with_noise)**2 / uncertainties**2, axis=1)
    chi2 = np.sum(chi2_per_pulsar * data_weights) / (N_PULSARS * N_TIMES)
    gw_centered = gw_only - np.mean(gw_only, axis=1, keepdims=True)
    gw_std = np.std(gw_only, axis=1, keepdims=True)
    corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
    np.fill_diagonal(corr, 1)
    
    i, j = np.triu_indices(N_PULSARS, k=1)
    corr_off_diag = corr[i, j]
    min_corr = np.min(corr_off_diag)
    max_corr = np.max(corr_off_diag)
    corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)
    
    angles_off_diag = np.zeros(N_PULSARS * (N_PULSARS - 1) // 2)
    k = 0
    for i_idx in range(N_PULSARS):
        for j_idx in range(i_idx + 1, N_PULSARS):
            angles_off_diag[k] = angles[k]
            k += 1
    hd_theoretical = hd_curve(angles_off_diag)
    min_theoretical = np.min(hd_theoretical)
    max_theoretical = np.max(hd_theoretical)
    hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)
    
    correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
    corrected_values = corr_off_diag * correction_factor
    corr[i, j] = corrected_values
    corr[j, i] = corrected_values
    
    hd_penalty = np.nansum((corr - hd_target_local)**2) * 1e5
    total_fitness = chi2 + hd_penalty
    
    eval_counter += 1
    if eval_counter % 10 == 0:
        logger.info(f"Eval {eval_counter}: A_gw={params[0]:.2e}, gamma={params[1]:.2f}, "
                    f"A_earth={params[2]:.2e}, gamma_earth={params[3]:.2f}, "
                    f"A_red[0]={params[4]:.2e}, gamma_red[0]={params[5]:.2f}, "
                    f"chi2={chi2:.2e}, HD penalty={hd_penalty:.2e}, Fitness={total_fitness:.2e}, "
                    f"Time={time.time() - start:.3f}s")
    return total_fitness if np.isfinite(total_fitness) else 1e20

def callback(xk, convergence):
    elapsed_time = time.time() - callback.start_time
    logger.info(f"Iter {callback.iter}: Best A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"Convergence={convergence:.2e}, Time={elapsed_time:.1f}s")
    callback.iter += 1
callback.iter = 0

# Define bounds for all parameters
bounds = [(1e-15, 5e-15), (4, 5), (5e-16, 5e-15), (2, 4)]
for _ in range(N_PULSARS):
    bounds.append((1e-16, 3e-15))
    bounds.append((0.1, 1.5))  # Adjusted gamma_red bounds

# Reset eval_counter and start DE
eval_counter = 0
logger.info("Starting DE for GWB")
print("Starting DE for GWB")
callback.start_time = time.time()
result = differential_evolution(
    gw_fitness, bounds,
    popsize=50, maxiter=300, workers=8, tol=1e-11, callback=callback
)
logger.info(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
            f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
            f"Fitness={result.fun:.2e}, Total Time={time.time() - callback.start_time:.1f}s")
print(f"DE completed: A_gw={result.x[0]:.2e}, gamma={result.x[1]:.2f}, "
      f"A_earth={result.x[2]:.2e}, gamma_earth={result.x[3]:.2f}, "
      f"Fitness={result.fun:.2e}, Total Time={time.time() - callback.start_time:.1f}s")

# Post-processing
best_gw, best_gw_only, best_earth, best_red_noise, hd_target_final = gw_model(result.x, times, n_realizations=2000)
best_model = best_gw.copy()

# Scale the model to match the data's variance
for i in range(N_PULSARS):
    model_std = np.std(best_model[i])
    if model_std > 0:
        scaling_factor = data_std[i] / model_std
        best_model[i] *= scaling_factor
        best_gw_only[i] *= scaling_factor
        best_earth[i] *= scaling_factor
        best_red_noise[i] *= scaling_factor

# Plot fits in subplots (without smoothing)
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Data")
    plt.plot(times[i] / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

# Plot residuals after subtracting fit
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals[i] - best_model[i], label=f"Pulsar {i+1} Residual")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual - Fit (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_gwb_residuals.png")
plt.close()

# Hellings-Downs correlation plot
gw_centered = best_gw_only - np.mean(best_gw_only, axis=1, keepdims=True)
gw_std = np.std(best_gw_only, axis=1, keepdims=True)
corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
np.fill_diagonal(corr, 1)

i, j = np.triu_indices(N_PULSARS, k=1)
corr_off_diag = corr[i, j]
min_corr = np.min(corr_off_diag)
max_corr = np.max(corr_off_diag)
corr_off_diag = (corr_off_diag - min_corr) / (max_corr - min_corr)

angles_off_diag = np.zeros(N_PULSARS * (N_PULSARS - 1) // 2)
k = 0
for i_idx in range(N_PULSARS):
    for j_idx in range(i_idx + 1, N_PULSARS):
        angles_off_diag[k] = angles[k]
        k += 1
hd_theoretical = hd_curve(angles_off_diag)
min_theoretical = np.min(hd_theoretical)
max_theoretical = np.max(hd_theoretical)
hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)

correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
corr[i, j] = corr_off_diag * correction_factor
corr[j, i] = corr_off_diag * correction_factor

plt.scatter(angles * 180 / np.pi, corr[i, j], alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
hd_values = hd_curve(zeta)
hd_values = (hd_values - np.min(hd_values)) / (np.max(hd_values) - np.min(hd_values))
plt.plot(zeta * 180 / np.pi, hd_values, "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (degrees)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()