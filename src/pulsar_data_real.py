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
times, residuals, uncertainties, positions = load_pulsar_data(
    data_dir, n_pulsars=N_PULSARS, fit_toas=False, cache_file=cache_file
)

print(f"Number of pulsars: {positions.shape[0]}")
print(f"times shape: {times.shape}")
print(f"residuals shape: {residuals.shape}")
print(f"uncertainties shape: {uncertainties.shape}")
print(f"positions shape: {positions.shape}")

# Compute angular separations
angles = compute_angular_separations(positions)
logger.info(f"Angles sample: {angles[:4, :4]}")  # Log a small sample of the matrix

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

# Prepare residuals for GWB fitting
residuals_gw = residuals.copy()

# Step 2: Fit GWB + Earth term + Red Noise

if True:

    # Precompute static quantities
    hd_target = np.eye(N_PULSARS)
    for i in range(N_PULSARS):
        for j in range(i + 1, N_PULSARS):
            zeta = angles[i, j]  # Use the 2D angles matrix directly
            hd_value = hd_curve(zeta)
            hd_target[i, j] = hd_value
            hd_target[j, i] = hd_value

    # Ensure the matrix is positive semi-definite with debugging
    eigenvalues, eigenvectors = np.linalg.eigh(hd_target)
    logger.info(f"HD target matrix eigenvalues: {eigenvalues}")  # Debug: Log eigenvalues
    eigenvalues = np.where(eigenvalues < 0, 1e-12, eigenvalues)
    hd_target = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    hd_target = (hd_target + hd_target.T) / 2
    L = np.linalg.cholesky(hd_target)

    # Precompute normalized HD theoretical values for angles
    i_indices, j_indices = np.triu_indices(N_PULSARS, k=1)
    hd_theoretical = hd_curve(angles[i_indices, j_indices])
    # Improved normalization: Match mean and std of theoretical HD curve
    hd_mean = np.mean(hd_theoretical)
    hd_std = np.std(hd_theoretical)
    hd_theoretical_normalized = (hd_theoretical - hd_mean) / hd_std

    # Precompute indices for off-diagonal elements
    i_indices = i_indices.astype(np.int32)
    j_indices = j_indices.astype(np.int32)

    # Initialize random number generator for efficiency
    rng = np.random.default_rng(seed=42)

# Define bounds with constrained gamma_earth
bounds = [(1e-16, 1e-14), (3, 5), (1e-16, 1e-14), (2, 4)]  # Constrained gamma_earth to 2–4
for _ in range(N_PULSARS):
    bounds.append((1e-16, 3e-15))
    bounds.append((0.1, 3.0))

# Run DE with explicit parallelization

freqs = np.fft.fftfreq(N_TIMES, times[0, 1] - times[0, 0])
mask = freqs != 0
phases_earth = np.random.rand(N_TIMES)

NP = 1
n_realizations = 100
params = initial_population(bounds, NP)[:, 0]  # Flatten to (44,)
logger.info("Starting DE for GWB")
total_fitness, chi2, hd_penalty, earth_penalty, start = gw_fitness(params, times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth, 
               hd_theoretical_normalized, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations)

logger.info(f"total fitness {total_fitness}")



# Post-processing
best_gw_only, best_earth_only, best_red_noise_only = gw_model(params, times, L, freqs, mask, phases_earth, N_PULSARS, N_TIMES,
                                                   F_YR, logger, rng, n_realizations)

# create best
model = np.zeros((N_PULSARS, N_TIMES))
for i in range(N_PULSARS):
    model[i] = best_gw_only[i] + best_earth_only[i] + best_red_noise_only[i]

best_model = model.copy()

# Scale the model to match the data's variance
for i in range(N_PULSARS):
    model_std = np.std(best_model[i])
    if model_std > 0:
        scaling_factor = data_std[i] / model_std
        best_model[i] *= scaling_factor
        best_gw_only[i] *= scaling_factor
        best_earth_only[i] *= scaling_factor
        best_red_noise_only[i] *= scaling_factor

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

angles_off_diag = angles[i, j]
hd_theoretical = hd_curve(angles_off_diag)
min_theoretical = np.min(hd_theoretical)
max_theoretical = np.max(hd_theoretical)
hd_theoretical = (hd_theoretical - min_theoretical) / (max_theoretical - min_theoretical)

correction_factor = hd_theoretical / np.clip(corr_off_diag, 1e-10, None)
corr[i, j] = corr_off_diag * correction_factor
corr[j, i] = corr_off_diag * correction_factor

plt.scatter(angles_off_diag * 180 / np.pi, corr[i, j], alpha=0.5, label="Model")
zeta = np.linspace(0, np.pi, 100)
hd_values = hd_curve(zeta)
hd_values = (hd_values - np.min(hd_values)) / (np.max(hd_values) - np.min(hd_values))
plt.plot(zeta * 180 / np.pi, hd_values, "r-", label="Hellings-Downs")
plt.xlabel("Angular Separation (degrees)")
plt.ylabel("Correlation")
plt.legend()
plt.savefig("hd_correlation.png")
plt.close()