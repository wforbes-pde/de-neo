import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy.optimize import differential_evolution

# Step 1: Load preprocessed data
data = np.load('microlensing_data.npz')
time = data['time']
flux = data['flux']
flux_err = data['flux_err']

# Inspect data
time_range = max(time) - min(time)
flux_max = max(flux)
print(f"Time range: {time_range:.2f} days")
print(f"Flux range: {min(flux):.4f} to {flux_max:.4f}")

# Step 2: Define chi-squared for PSPL model
def chi2_pspl(params, time, flux, flux_err):
    t_0, u_0, t_E, F_s, F_b = params
    
    # Compute magnification
    u = np.sqrt(u_0**2 + ((time - t_0) / t_E)**2)
    magnification = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    
    # Model flux
    model_flux = F_s * magnification + F_b
    
    # Check for invalid flux
    if np.any(model_flux <= 0) or np.any(np.isnan(model_flux)):
        return np.inf
    
    # Chi-squared
    chi2_value = np.sum(((flux - model_flux) / flux_err)**2)
    if np.isnan(chi2_value):
        return np.inf
    return chi2_value

# Step 3: Set up DE optimization for PSPL
bounds_pspl = [
    (-time_range/4, time_range/4),  # t_0
    (0, 1),                        # u_0
    (1, time_range*1.5),           # t_E
    (0, flux_max*1.5),             # F_s
    (0, flux_max*1.5)              # F_b
]

print("Running DE for PSPL model...")
result_pspl = differential_evolution(
    func=lambda params: chi2_pspl(params, time, flux, flux_err),
    bounds=bounds_pspl,
    strategy='best1bin',
    maxiter=500,  # Fewer iterations (simpler model)
    popsize=15,
    tol=0.01,
    disp=True
)

# Extract best-fit parameters
best_params_pspl = result_pspl.x
best_chi2_pspl = result_pspl.fun
param_names_pspl = ['t_0', 'u_0', 't_E', 'F_s', 'F_b']
print("Best-fit PSPL parameters:")
for name, value in zip(param_names_pspl, best_params_pspl):
    print(f"{name}: {value:.4f}")
print(f"PSPL chi-squared: {best_chi2_pspl:.2f}")

# Step 4: Compute PSPL model for plotting
u = np.sqrt(best_params_pspl[1]**2 + ((time - best_params_pspl[0]) / best_params_pspl[2])**2)
magnification_pspl = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
model_flux_pspl = best_params_pspl[3] * magnification_pspl + best_params_pspl[4]

# Step 5: Compare with binary lens model
# Load binary lens results
binary_data = np.load('best_fit.npz')
best_params_binary = binary_data['params']
best_chi2_binary = binary_data['chi2']

from MulensModel import Model
model_binary = Model(parameters={
    't_0': best_params_binary[0],
    'u_0': best_params_binary[1],
    't_E': best_params_binary[2],
    'q': best_params_binary[3],
    's': best_params_binary[4],
    'alpha': best_params_binary[5]
})
magnification_binary = model_binary.get_magnification(time)
model_flux_binary = best_params_binary[6] * magnification_binary + best_params_binary[7] + best_params_binary[8]

# Step 6: Plot comparison
plt.errorbar(time, flux, yerr=flux_err, fmt='o', ms=3, label='OGLE Data')
plt.plot(time, model_flux_binary, 'r-', label=f'Binary Lens (χ²={best_chi2_binary:.1f})')
plt.plot(time, model_flux_pspl, 'b--', label=f'PSPL (χ²={best_chi2_pspl:.1f})')
plt.xlabel('Time (HJD - 2450000 - t_peak)')
plt.ylabel('Flux (arbitrary units)')
plt.title('Model Comparison: OGLE-2023-BLG-0136')
plt.gca().invert_yaxis()
plt.legend()
plt.savefig('model_comparison.png')
print("Comparison plot saved to 'model_comparison.png'")
plt.close()

# Step 7: Save PSPL results
np.savez('best_fit_pspl.npz', params=best_params_pspl, chi2=best_chi2_pspl, param_names=param_names_pspl)
print("PSPL parameters saved to 'best_fit_pspl.npz'")

# Step 8: Evaluate fit quality
N = len(time)
dof_binary = N - 9
dof_pspl = N - 5
print(f"Data points: {N}")
print(f"Binary Lens: DOF={dof_binary}, Reduced χ²={best_chi2_binary/dof_binary:.2f}")
print(f"PSPL: DOF={dof_pspl}, Reduced χ²={best_chi2_pspl/dof_pspl:.2f}")