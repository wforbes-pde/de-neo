import numpy as np
import logging
import pickle
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_filter1d

from pulsar_data import load_pulsar_data, interpolate_residuals, compute_angular_separations
from pulsar_data import hd_curve, gw_fitness, gw_model, to_shared_array, callback, initial_population
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

N_PULSARS = 20
N_TIMES = 1000
F_YR = 1 / (365.25 * 86400)

# Load the data
data_dir = "/home/wesley/data"
cache_file = "pulsar_data_cache.pkl"
# After loading data
times, residuals, uncertainties, positions = load_pulsar_data(
    data_dir, n_pulsars=N_PULSARS, fit_toas=True, cache_file=cache_file,
)

# Plot pre-glitch residuals
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Pre-Glitch")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_pre_glitch_residuals.png")
plt.close()

# Proceed with interpolation and GWB fitting
times, residuals, uncertainties = interpolate_residuals(times, residuals, uncertainties, N_TIMES)

# Plot post-glitch residuals
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Post-Glitch")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_post_glitch_residuals.png")
plt.close()






print(f"Number of pulsars: {len(positions)}")
print(f"times shape: {len(times)}")
print(f"residuals shape: {len(residuals)}")
print(f"uncertainties shape: {len(uncertainties)}")
print(f"positions shape: {len(positions)}")

# Log residuals stats
logger.info(f"Residuals mean: {np.nanmean(residuals, axis=1)[:3]}")
logger.info(f"Residuals std: {np.nanstd(residuals, axis=1)[:3]}")
logger.info(f"Residuals sample: {residuals[0][:5]}")

# Compute angular separations
angles = compute_angular_separations(positions)
logger.info(f"Angles sample: {angles[:4, :4]}")

# Interpolate residuals
times, residuals, uncertainties = interpolate_residuals(times, residuals, uncertainties, N_TIMES)

# Plot raw residuals
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals[i], label=f"Pulsar {i+1} Raw")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_raw_residuals.png")
plt.close()

# Estimate data noise level
data_std = np.array([np.std(residuals[i][~np.isnan(residuals[i])]) for i in range(N_PULSARS)])
print(f"Data standard deviations: {data_std[:3]}")
print(f"Uncertainties: {uncertainties[:3, 0]}")

# Compute weights based on data length
data_weights = np.array([np.sum(~np.isnan(residuals[i])) for i in range(N_PULSARS)], dtype=float)
data_weights /= data_weights.sum()

# Center residuals
residuals_gw = residuals - np.nanmean(residuals, axis=1, keepdims=True)

# Precompute static quantities
hd_target = np.eye(N_PULSARS)
for i in range(N_PULSARS):
    for j in range(i + 1, N_PULSARS):
        zeta = angles[i, j]
        hd_value = hd_curve(zeta)
        hd_target[i, j] = hd_value
        hd_target[j, i] = hd_value

eigenvalues, eigenvectors = np.linalg.eigh(hd_target)
logger.info(f"HD target matrix eigenvalues: {eigenvalues}")
eigenvalues = np.where(eigenvalues < 0, 1e-12, eigenvalues)
hd_target = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
hd_target = (hd_target + hd_target.T) / 2
L = np.linalg.cholesky(hd_target)

i_indices, j_indices = np.triu_indices(N_PULSARS, k=1)
hd_theoretical = hd_curve(angles[i_indices, j_indices])

i_indices = i_indices.astype(np.int32)
j_indices = j_indices.astype(np.int32)

rng = np.random.default_rng(seed=42)

# Define bounds
bounds = [(5e-16, 5e-14), (3.5, 4.5), (5e-16, 5e-14), (2, 4)]
for _ in range(N_PULSARS):
    bounds.append((1e-16, 3e-15))
    bounds.append((0.1, 3.0))

# Initialize population
NPs=50
maxiters = 5
initpop = initial_population(bounds,NP=NPs)
# Transpose to match expected shape (n_pop, n_params)
initpop = initpop.T
# initpop[:, 0] = rng.uniform(1e-15, 3e-15, initpop.shape[0])  # A_gw
# initpop[:, 1] = rng.uniform(4.0, 4.5, initpop.shape[0])      # gamma
# initpop[:, 2] = rng.uniform(1e-15, 3e-15, initpop.shape[0])  # A_earth
# initpop[:, 3] = rng.uniform(2.5, 3.5, initpop.shape[0])      # gamma_earth

# Run DE
n_realizations = 100
freqs = np.fft.fftfreq(N_TIMES, times[0, 1] - times[0, 0])
mask = freqs != 0
logger.info(f"freqs shape: {freqs.shape}, mask sum: {np.sum(mask)}")
phases_earth = np.random.rand(N_TIMES)

# Initialize callback attributes
callback.start_time = time.time()
callback.iter = 0
gw_fitness.eval_count = 0
callback.args = (times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth,
                 hd_theoretical, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations)

logger.info(f"Starting DE for GWB with popsize={NPs}, maxiter={maxiters}")
result = differential_evolution(
    gw_fitness,
    bounds=bounds,
    args=(times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth,
          hd_theoretical, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations),
    popsize=NPs,
    maxiter=maxiters,
    workers=6,
    tol=1e-4,
    callback=lambda xk, convergence: callback(xk, convergence, logger),
    disp=True,
    seed=42,
    polish=True,
    init=initpop
)

# Extract results
params = result.x
total_fitness, chi2, hd_penalty, earth_penalty, start = gw_fitness(
    params, times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth,
    hd_theoretical, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations,
    return_full=True
)
logger.info(f"Final fitness: {total_fitness}, chi2: {chi2}, hd_penalty: {hd_penalty}, earth_penalty: {earth_penalty}")

# Post-processing
best_gw_only, best_earth_only, best_red_noise_only = gw_model(params, times, L, freqs, mask, phases_earth, N_PULSARS, N_TIMES,
                                                             F_YR, logger, rng, n_realizations)

# Scale signals
for i in range(N_PULSARS):
    target_std = data_std[i] / 5  # Lower scaling factor
    for signal in [best_gw_only, best_earth_only, best_red_noise_only]:
        signal_std = np.std(signal[i])
        if signal_std > 0:
            signal[i] *= target_std / signal_std

# Create best model
best_model = best_gw_only + best_earth_only + best_red_noise_only

# Center model
best_model = best_model - np.mean(best_model, axis=1, keepdims=True)

# Plot fits
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals_gw[i], label=f"Pulsar {i+1} Data")
    plt.plot(times[i] / (365.25 * 86400), best_model[i], "--", label=f"Pulsar {i+1} Fit")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_gwb_fit.png")
plt.close()

# Plot residuals
plt.figure(figsize=(12, 8))
for i in range(min(3, N_PULSARS)):
    plt.subplot(3, 1, i+1)
    plt.plot(times[i] / (365.25 * 86400), residuals_gw[i] - best_model[i], label=f"Pulsar {i+1} Residual")
    plt.xlabel("Time (yr)")
    plt.ylabel("Residual - Fit (s)")
    plt.legend()
plt.tight_layout()
plt.savefig("pulsar_gwb_residuals.png")
plt.close()

# Hellings-Downs correlation
gw_centered = best_gw_only - np.mean(best_gw_only, axis=1, keepdims=True)
gw_std = np.std(best_gw_only, axis=1, keepdims=True) + 1e-12
corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
corr_off_diag = corr[i_indices, j_indices]
angles_off_diag = angles[i_indices, j_indices]

plt.figure(figsize=(8, 6))
plt.scatter(angles_off_diag * 180 / np.pi, corr_off_diag, alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
hd_values = hd_curve(zeta)
plt.plot(zeta * 180 / np.pi, hd_values, "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (degrees)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()



#n_realizations = 100
# NP = 1
# params = initial_population(bounds, NP)[:, 0]  # Flatten to (44,)
# logger.info("Starting DE for GWB")
# total_fitness, chi2, hd_penalty, earth_penalty, start = gw_fitness(params, times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth, 
#                hd_theoretical_normalized, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations)

# logger.info(f"total fitness {total_fitness}")

