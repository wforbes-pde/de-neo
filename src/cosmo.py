import numpy as np
import pandas as pd
import GPy
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import logging
import os
from datetime import datetime
from io import StringIO
from sklearn.preprocessing import StandardScaler
import time

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Running cosmo.py at {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')}")

# Step 1: Retrieve Real SMF Data
def fetch_smf_data(url='https://www.astro.ljmu.ac.uk/~ikb/research/data/gsmf-B12.txt', local_file='gsmf-B12.txt'):
    logger.info(f"Attempting to fetch SMF data from {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = pd.read_csv(StringIO(response.text), sep=r'\s+', comment='#', 
                          names=['logMstar', 'bin_width', 'Phi', 'e_Phi', 'N'])
        log_mstar = data['logMstar'].values
        phi = data['Phi'].values  # 10^-3 Mpc^-3 dex^-1
        phi_err = data['e_Phi'].values
        logger.info("Successfully retrieved SMF data from URL")
        return log_mstar, phi, phi_err
    except Exception as e:
        logger.warning(f"Failed to fetch data from URL: {e}")
        if os.path.exists(local_file):
            logger.info(f"Loading local SMF data from {local_file}")
            data = pd.read_csv(local_file, sep=r'\s+', comment='#', 
                              names=['logMstar', 'bin_width', 'Phi', 'e_Phi', 'N'])
            log_mstar = data['logMstar'].values
            phi = data['Phi'].values
            phi_err = data['e_Phi'].values
            return log_mstar, phi, phi_err
        logger.error(f"No local file found at {local_file}. Using mock SMF.")
        log_mstar = np.linspace(6.25, 11.90, 27)
        phi = 7.4e-3 * (10**log_mstar / 10**10.6)**(-1.35) * np.exp(-(10**log_mstar / 10**10.6))
        phi_err = 0.1 * phi
        return log_mstar, phi, phi_err

# Step 2: Generate Training Data
def generate_training_data(n_samples=1000):
    logger.info(f"Generating {n_samples} training samples")
    np.random.seed(42)
    param_bounds = [
        (0.001, 0.7), (0.01, 4.0), (0.001, 0.7), (0.05, 0.8), (0.3, 2.0),
        (0.001, 0.3), (0.01, 1.2), (0.005, 0.6), (0.01, 1.5), (0.001, 0.4)
    ]
    params = np.random.uniform(
        [b[0] for b in param_bounds],
        [b[1] for b in param_bounds],
        (n_samples, len(param_bounds))
    )
    scaler_params = StandardScaler()
    params_scaled = scaler_params.fit_transform(params)
    
    log_mstar, _, _ = fetch_smf_data()
    smf_predictions = np.zeros((n_samples, len(log_mstar)))
    for i in range(n_samples):
        p = params[i]
        log_mstar0 = 10.6 + p[0] * 1.2 - p[3] * 0.6
        alpha_low = -0.5 - p[1] * 0.8 + p[4] * 0.4
        alpha_high = -1.35 - p[1] * 0.5 + p[4] * 0.3
        phi_star = 7.4e-3 * (1 + p[2] * 12.0) * (1 + p[5] * 2.5)
        m_ratio = 10**log_mstar / 10**log_mstar0
        smf = phi_star * np.where(log_mstar < 9.0,
                                 m_ratio**alpha_low,
                                 m_ratio**alpha_high) * \
              np.exp(-m_ratio)
        smf *= (1 + p[6] * 0.3 * np.sin(p[7] * log_mstar)) * \
               (1 + p[8] * (log_mstar - 6.0) / 8.0) * \
               (1 - p[9] * (log_mstar - 6.0) / 5.0)
        smf_predictions[i] = np.clip(smf, 1e-6, 150.0)
    
    scaler_smf = StandardScaler()
    smf_scaled = scaler_smf.fit_transform(smf_predictions)
    return params, smf_predictions, log_mstar, scaler_params, scaler_smf

# Step 3: Build Gaussian Process Emulator
def build_emulator(params, smf_predictions, scaler_params, scaler_smf):
    logger.info("Building Gaussian process emulator")
    params_scaled = scaler_params.transform(params)
    smf_scaled = scaler_smf.transform(smf_predictions)
    
    kernel = GPy.kern.RBF(input_dim=params.shape[1], variance=1.0, lengthscale=1.0, ARD=True)
    model = GPy.models.GPRegression(params_scaled, smf_scaled, kernel, noise_var=1e-4)
    model.kern.variance.constrain_bounded(0.1, 10.0)
    model.kern.lengthscale.constrain_bounded(0.1, 5.0)
    
    start_time = time.time()
    timeout = 300  # 5 minutes
    try:
        logger.info("Starting GP optimization with restarts")
        model.optimize_restarts(num_restarts=3, verbose=True, optimizer='lbfgsb', max_iters=500)
        logger.info(f"GP optimization with restarts completed in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.warning(f"optimize_restarts failed: {e}. Falling back to single optimization.")
        model.optimize(optimizer='lbfgsb', max_iters=500, messages=True)
        logger.info(f"Single GP optimization completed in {time.time() - start_time:.2f} seconds")
    if time.time() - start_time > timeout:
        logger.warning("GP optimization timed out after 5 minutes. Proceeding with current model.")
    
    # Validate emulator (overall)
    idx_test = np.random.choice(len(params), size=50, replace=False)
    params_test = params_scaled[idx_test]
    smf_test = smf_scaled[idx_test]
    smf_pred, _ = model.predict(params_test)
    mse = np.mean((smf_pred - smf_test)**2)
    logger.info(f"Emulator validation MSE (overall): {mse}")
    
    # Validate high-mass region (logMstar > 9)
    log_mstar, _, _ = fetch_smf_data()
    high_mass_indices = log_mstar > 9.0
    smf_high_mass = smf_predictions[:, high_mass_indices]  # Unscaled true values
    smf_pred_scaled, _ = model.predict(params_scaled)  # Predict for all bins
    smf_pred = scaler_smf.inverse_transform(smf_pred_scaled)  # Inverse transform all bins
    smf_high_pred = smf_pred[:, high_mass_indices]  # Subset to high-mass bins
    mse_high = np.mean((smf_high_pred - smf_high_mass)**2)
    logger.info(f"Emulator validation MSE (high mass, logMstar > 9): {mse_high}")
    
    return model, scaler_smf

# Step 4: Define Objective Function for DE
def objective_function(params, emulator, scaler_params, scaler_smf, log_mstar, phi_obs, phi_err):
    params_scaled = scaler_params.transform(params.reshape(1, -1))
    phi_pred_scaled, _ = emulator.predict(params_scaled)
    phi_pred = scaler_smf.inverse_transform(phi_pred_scaled).flatten()
    phi_pred = np.clip(phi_pred, 1e-6, 150.0)
    chi2 = np.sum(((phi_pred - phi_obs) / phi_err)**2)
    key_indices = [3, 9, 19, 25]  # logMstar = 7.3, 8.5, 10.5, 11.7
    logger.debug(f"Predicted Phi at key points: {phi_pred[key_indices]}")
    logger.debug(f"Observed Phi at key points: {phi_obs[key_indices]}")
    return chi2

# Step 5: Main Optimization Routine
def main():
    log_mstar, phi_obs, phi_err = fetch_smf_data()
    
    params, smf_predictions, log_mstar_sim, scaler_params, scaler_smf = generate_training_data(n_samples=1000)
    assert np.allclose(log_mstar, log_mstar_sim), "SMF bins must match"
    
    emulator, scaler_smf = build_emulator(params, smf_predictions, scaler_params, scaler_smf)
    
    param_bounds = [
        (0.001, 0.7), (0.01, 4.0), (0.001, 0.7), (0.05, 0.8), (0.3, 2.0),
        (0.001, 0.3), (0.01, 1.2), (0.005, 0.6), (0.01, 1.5), (0.001, 0.4)
    ]
    
    logger.info("Starting differential evolution optimization")
    result = differential_evolution(
        func=objective_function,
        args=(emulator, scaler_params, scaler_smf, log_mstar, phi_obs, phi_err),
        bounds=param_bounds,
        strategy='best2bin',
        maxiter=2000,
        popsize=50,
        tol=0.005,
        seed=42,
        disp=True
    )
    
    logger.info(f"Optimization successful: {result.success}")
    logger.info(f"Best parameters: {result.x}")
    logger.info(f"Chi-squared: {result.fun}")
    
    best_params = result.x.reshape(1, -1)
    phi_pred_scaled, _ = emulator.predict(scaler_params.transform(best_params))
    phi_pred = scaler_smf.inverse_transform(phi_pred_scaled).flatten()
    
    plt.errorbar(log_mstar, phi_obs, yerr=phi_err, fmt='o', label='Observed SMF')
    plt.plot(log_mstar, phi_pred, label='Predicted SMF')
    plt.xlabel('log(Mstar/Msun)')
    plt.ylabel('Phi (10^-3 Mpc^-3 dex^-1)')
    plt.yscale('log')
    plt.legend()
    plt.savefig('smf_comparison.png')
    plt.close()
    logger.info("Plot saved to smf_comparison.png")

if __name__ == "__main__":
    main()