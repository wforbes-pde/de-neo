import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter
import pint.logging
import os
import re
from astropy import units as u
from astropy.coordinates import SkyCoord
from multiprocessing import Pool, cpu_count
import logging
from functools import partial
import pickle
import time
import multiprocessing as mp

# Suppress PINT logging to avoid clutter
pint.logging.setup(level="WARNING")

# Set up logging for parallel processes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_pulsar(args, fit_toas=True):
    psr_name, par_file, tim_file = args
    try:
        logging.info(f"Loading data for {psr_name} (par: {par_file}, tim: {tim_file})")
        model = get_model(par_file)
        toas = get_TOAs(tim_file, model=model)

        if fit_toas:
            fitter = WLSFitter(toas=toas, model=model)
            fitter.fit_toas()
        else:
            from pint.residuals import Residuals
            fitter = WLSFitter(toas=toas, model=model)
            residuals_obj = Residuals(toas=toas, model=model)
            fitter.resids = residuals_obj

        logging.info(f"Model components for {psr_name}: {fitter.model.components.keys()}")
        logging.info(f"Astrometry parameters: {fitter.model.get_params_of_type('astrometry')}")

        mjds = toas.get_mjds().value
        mjds_sorted = np.sort(mjds)
        times_i = (mjds - mjds_sorted[0]) * 86400.0
        obs_span = (mjds_sorted[-1] - mjds_sorted[0]) * 86400.0

        residuals_i = fitter.resids.time_resids.to("s").value
        uncertainties_i = toas.get_errors().to("s").value

        if "AstrometryEquatorial" in fitter.model.components:
            logging.info(f"Using AstrometryEquatorial for {psr_name}")
            ra = fitter.model.RAJ.quantity.to("rad").value
            dec = fitter.model.DECJ.quantity.to("rad").value
        elif "AstrometryEcliptic" in fitter.model.components:
            logging.info(f"Using AstrometryEcliptic for {psr_name}")
            elong = fitter.model.ELONG.value
            elat = fitter.model.ELAT.value
            elong = elong * u.deg
            elat = elat * u.deg
            coord = SkyCoord(lon=elong, lat=elat, frame='barycentrictrueecliptic')
            equatorial = coord.transform_to('icrs')
            ra = equatorial.ra.to('rad').value
            dec = equatorial.dec.to('rad').value
            logging.info(f"Converted RA, Dec for {psr_name}: {ra} rad, {dec} rad")
        else:
            raise ValueError(f"No recognized astrometry component for {psr_name}.")

        efac_param = getattr(fitter.model, "EFAC1", None)
        efac = efac_param.value if efac_param is not None else 1.0
        equad_param = getattr(fitter.model, "EQUAD1", None)
        equad = equad_param.value if equad_param is not None else 0.0
        mean_uncertainty = np.mean(uncertainties_i)
        noise_level = efac * mean_uncertainty + equad
        logging.info(f"Noise level components for {psr_name}: EFAC={efac}, EQUAD={equad}, Mean Uncertainty={mean_uncertainty:.2e} s, Noise Level={noise_level:.2e} s")

        return {
            "name": psr_name,
            "times": times_i,
            "residuals": residuals_i,
            "uncertainties": uncertainties_i,
            "position": (ra, dec),
            "obs_span": obs_span,
            "noise_level": noise_level
        }
    except Exception as e:
        logging.error(f"Error loading data for {psr_name}: {e}")
        return None

def load_pulsar_data(data_dir, n_pulsars=20, use_wideband=False, n_jobs=None, fit_toas=True, cache_file="pulsar_data_cache.pkl"):
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data["times"], cached_data["residuals"], cached_data["uncertainties"], cached_data["positions"]

    data_type = "wideband" if use_wideband else "narrowband"
    data_path = os.path.join(data_dir, data_type)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} not found. Ensure the NANOGrav 15-year dataset is correctly structured.")

    par_path = os.path.join(data_path, "par")
    tim_path = os.path.join(data_path, "tim")

    if not os.path.exists(par_path):
        raise FileNotFoundError(f"Par directory {par_path} not found.")
    if not os.path.exists(tim_path):
        raise FileNotFoundError(f"Tim directory {tim_path} not found.")

    par_files = [f for f in os.listdir(par_path) if f.endswith(".par")]
    tim_files = [f for f in os.listdir(tim_path) if f.endswith(".tim")]

    print(f"Found {len(par_files)} .par files in {par_path}: {par_files[:5] if par_files else 'None'}")
    print(f"Found {len(tim_files)} .tim files in {tim_path}: {tim_files[:5] if tim_files else 'None'}")

    pulsar_files = []
    for par_file in par_files:
        par_base = os.path.splitext(par_file)[0]
        match = re.match(r"(J\d{4}[+-]\d{4})", par_file)
        if not match:
            print(f"Skipping {par_file}: Does not match expected pulsar name format (e.g., J0030+0451).")
            continue
        psr_name = match.group(1)
        tim_file = f"{par_base}.tim"
        if tim_file in tim_files:
            pulsar_files.append((psr_name, os.path.join(par_path, par_file), os.path.join(tim_path, tim_file)))
        else:
            print(f"No matching .tim file found for {psr_name} (par: {par_file}), expected {tim_file}, skipping.")

    if not pulsar_files:
        print("Available .tim files:", tim_files)
        raise ValueError(f"No matching .par/.tim pairs found in {data_path}. Check file naming and directory structure.")

    print(f"Found {len(pulsar_files)} matching .par/.tim pairs.")

    if len(pulsar_files) < n_pulsars:
        raise ValueError(f"Found only {len(pulsar_files)} pulsars, but {n_pulsars} are requested.")

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    print(f"Using {n_jobs} parallel jobs to process {n_pulsars} pulsars.")

    process_func = partial(process_single_pulsar, fit_toas=fit_toas)
    with Pool(processes=n_jobs) as pool:
        pulsar_data = pool.map(process_func, pulsar_files[:n_pulsars])

    pulsar_data = [p for p in pulsar_data if p is not None]

    if len(pulsar_data) < n_pulsars:
        raise ValueError(f"Successfully loaded only {len(pulsar_data)} pulsars, but {n_pulsars} are requested.")

    pulsar_data.sort(key=lambda x: (-x["obs_span"], x["noise_level"]))

    times_list = []
    residuals_list = []
    uncertainties_list = []
    positions = []

    for pulsar in pulsar_data[:n_pulsars]:
        times_list.append(pulsar["times"])
        residuals_list.append(pulsar["residuals"])
        uncertainties_list.append(pulsar["uncertainties"])
        positions.append(pulsar["position"])

    # Convert to arrays, padding with NaNs for irregular lengths
    max_n_times = max(len(t) for t in times_list)
    times = np.full((n_pulsars, max_n_times), np.nan)  # Pad times
    residuals = np.full((n_pulsars, max_n_times), np.nan)
    uncertainties = np.full((n_pulsars, max_n_times), np.nan)

    for i in range(n_pulsars):
        n_times_i = len(times_list[i])
        times[i, :n_times_i] = times_list[i]  # Pad times
        residuals[i, :n_times_i] = residuals_list[i]
        uncertainties[i, :n_times_i] = uncertainties_list[i]

    positions = np.array(positions)

    # Cache the results
    cache_data = {
        "times": times,  # Now a padded numpy array
        "residuals": residuals,
        "uncertainties": uncertainties,
        "positions": positions
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"Saved data to cache file {cache_file}")

    print("Selected pulsars:")
    for i, pulsar in enumerate(pulsar_data):
        print(f"Pulsar {i+1}: {pulsar['name']}, Obs Span: {pulsar['obs_span']/(365.25*86400):.2f} years, "
              f"Noise Level: {pulsar['noise_level']:.2e} s")

    return times, residuals, uncertainties, positions

def interpolate_residuals(times, residuals, uncertainties, N_times):
    n_pulsars = residuals.shape[0]
    times_new = np.zeros((n_pulsars, N_times))
    residuals_new = np.zeros((n_pulsars, N_times))
    uncertainties_new = np.zeros((n_pulsars, N_times))

    for i in range(n_pulsars):
        mask = ~np.isnan(residuals[i])
        t_i = times[i][mask]
        r_i = residuals[i][mask]
        u_i = uncertainties[i][mask]

        assert len(t_i) == len(r_i) == len(u_i), f"Length mismatch for pulsar {i}: times={len(t_i)}, residuals={len(r_i)}, uncertainties={len(u_i)}"

        t_new_i = np.linspace(t_i.min(), t_i.max(), N_times)
        residuals_new[i] = np.interp(t_new_i, t_i, r_i)
        uncertainties_new[i] = np.interp(t_new_i, t_i, u_i)
        times_new[i] = t_new_i

    return times_new, residuals_new, uncertainties_new


def compute_angular_separations(positions):
    """
    Compute angular separations between pulsars.
    positions: Array [n_pulsars, 2], (RA, Dec) in radians.
    Returns:
        angles: Array [n_pulsars, n_pulsars], angular separations in radians.
    """
    n_pulsars = len(positions)
    angles = np.zeros((n_pulsars, n_pulsars))
    for i in range(n_pulsars):
        for j in range(n_pulsars):
            if i == j:
                angles[i, j] = 0
            else:
                ra1, dec1 = positions[i]
                ra2, dec2 = positions[j]
                cos_zeta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
                cos_zeta = np.clip(cos_zeta, -1, 1)  # Avoid numerical errors
                angles[i, j] = np.arccos(cos_zeta)
    return angles

def hd_curve(zeta):
    x = (1 - np.cos(zeta)) / 2
    x = np.clip(x, 1e-10, None)
    return 1.5 * x * np.log(x) - 0.25 * x + 0.5




# Convert arrays to shared memory to reduce pickling overhead
def to_shared_array(arr):
    if arr.dtype == np.bool:
        typecode = 'b'  # Use bytes for boolean arrays (1 byte)
    elif arr.dtype in (np.float64, np.float32):
        typecode = 'd'  # Use double for floating-point arrays (8 bytes)
    elif arr.dtype == np.int64:
        typecode = 'q'  # Use long long for 64-bit integers (8 bytes)
    elif arr.dtype == np.int32:
        typecode = 'i'  # Use int for 32-bit integers (4 bytes)
    else:
        raise ValueError(f"Unsupported dtype {arr.dtype} for shared array")

    shared_arr = mp.RawArray(typecode, arr.size)
    shared_arr_np = np.frombuffer(shared_arr, dtype=arr.dtype).reshape(arr.shape)
    shared_arr_np[:] = arr[:]
    return shared_arr, shared_arr_np


def initial_population(bounds, NP):
    
    m = len(bounds)
    params = np.zeros((m, NP))
    for c in np.arange(0,NP):
        for e in np.arange(0,m):
            current_tuple = bounds[e]
            upper = current_tuple[0]
            lower = current_tuple[1]
            params[e,c] = np.random.uniform(low=lower, high=upper)

    return params




def callback(xk, convergence, logger):
    elapsed_time = time.time() - callback.start_time
    fitness, chi2, hd_penalty, earth_penalty, _ = gw_fitness(
        xk, *callback.args, return_full=True
    )
    gw_model_args = (xk, callback.args[0], callback.args[5], callback.args[6], callback.args[7], callback.args[8],
                     callback.args[12], callback.args[13], callback.args[14], callback.args[15], callback.args[16], callback.args[17])
    residuals_diff = callback.args[1] - gw_model(*gw_model_args)[0]
    logger.info(f"Iteration {callback.iter + 1}/3: A_gw={xk[0]:.2e}, gamma={xk[1]:.2f}, "
                f"A_earth={xk[2]:.2e}, gamma_earth={xk[3]:.2f}, "
                f"Fitness={fitness:.2e}, chi2={chi2:.2e}, hd_penalty={hd_penalty:.2e}, "
                f"Residuals_diff_std={np.nanstd(residuals_diff):.2e}, "
                f"Convergence={convergence:.2e}, Time={elapsed_time:.1f}s")
    callback.iter += 1
    return False

def gw_model(params, times, L, freqs, mask, phases_earth, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations):
    A_gw, gamma, A_earth, gamma_earth = params[:4]
    red_noise_params = params[4:]
    red_noise_params = red_noise_params.reshape((N_PULSARS, 2))
    A_red = red_noise_params[:, 0]
    gamma_red = red_noise_params[:, 1]
    
    random_signals = rng.normal(0, 1, (n_realizations, N_PULSARS, N_TIMES))
    random_signals = np.tensordot(L, random_signals, axes=(1, 1)).transpose(1, 0, 2)
    
    gw_only = np.zeros((N_PULSARS, N_TIMES))
    earth_only = np.zeros((N_PULSARS, N_TIMES))
    red_noise_only = np.zeros((N_PULSARS, N_TIMES))
    
    freqs_masked = freqs[mask]
    
    signals = random_signals.reshape(n_realizations * N_PULSARS, N_TIMES)
    signals_fft = np.fft.fft(signals, axis=1)[:, mask]
    psd_gw = (A_gw**2 / 12 / np.pi**2) * (np.abs(freqs_masked) / F_YR)**(-gamma) * F_YR**3
    psd_gw = np.maximum(psd_gw, 1e-9)
    signals_fft *= np.sqrt(psd_gw)
    signals_fft_full = np.zeros((n_realizations * N_PULSARS, N_TIMES), dtype=complex)
    signals_fft_full[:, mask] = signals_fft
    signals = np.fft.ifft(signals_fft_full, n=N_TIMES, axis=1).real
    signals = signals.reshape(n_realizations, N_PULSARS, N_TIMES)
    gw_only = np.mean(signals, axis=0)
    
    for i in range(N_PULSARS):
        signal = rng.normal(0, 1, N_TIMES)
        signal_fft_full = np.fft.fft(signal)
        signal_fft = signal_fft_full[mask]
        psd_earth = (A_earth**2 / 12 / np.pi**2) * (np.abs(freqs_masked) / F_YR)**(-gamma_earth) * F_YR**3
        psd_earth = np.maximum(psd_earth, 1e-9)
        signal_fft = signal_fft * np.sqrt(psd_earth) * np.exp(1j * phases_earth[mask])
        signal_fft_full[mask] = signal_fft
        signal = np.fft.ifft(signal_fft_full, n=N_TIMES).real
        earth_only[i] = signal
        
        signal = rng.normal(0, 1, N_TIMES)
        signal_fft_full = np.fft.fft(signal)
        signal_fft = signal_fft_full[mask]
        psd_red = (A_red[i]**2 / 12 / np.pi**2) * (np.abs(freqs_masked) / F_YR)**(-gamma_red[i]) * F_YR**3
        psd_red = np.maximum(psd_red, 1e-9)
        signal_fft = signal_fft * np.sqrt(psd_red)
        signal_fft_full[mask] = signal_fft
        signal = np.fft.ifft(signal_fft_full, n=N_TIMES).real
        red_noise_only[i] = signal
    
    if np.any(np.std(gw_only, axis=1) < 1e-12):
        logger.debug(f"Low gw_only std: {np.std(gw_only, axis=1)}")
    
    return gw_only, earth_only, red_noise_only

def gw_fitness(params, times, residuals_gw, uncertainties, data_weights, data_std, L, freqs, mask, phases_earth, 
               hd_theoretical, i_indices, j_indices, N_PULSARS, N_TIMES, F_YR, logger, rng, n_realizations,
               return_full=False):
    gw_fitness.eval_count = getattr(gw_fitness, 'eval_count', 0) + 1
    if gw_fitness.eval_count <= 5:
        logger.info(f"Fitness evaluation {gw_fitness.eval_count}")
    
    start = time.time()
    A_gw, gamma, A_earth, gamma_earth = params[:4]
    gw_only, earth_only, red_noise_only = gw_model(params, times, L, freqs, mask, phases_earth, N_PULSARS, N_TIMES,
                                                   F_YR, logger, rng, n_realizations)
    
    if gw_fitness.eval_count <= 5:
        logger.info(f"gw_only shape: {gw_only.shape}")
        logger.info(f"earth_only shape: {earth_only.shape}")
        logger.info(f"red_noise_only shape: {red_noise_only.shape}")
        logger.info(f"GWB signal std per pulsar: {np.std(gw_only, axis=1)}")
        logger.debug(f"Raw gw_only sample: {gw_only[0][:5]}")
    
    model = gw_only + earth_only + red_noise_only
    model = model - np.mean(model, axis=1, keepdims=True)

    white_noise = rng.normal(0, data_std[:, np.newaxis] / 8, (N_PULSARS, N_TIMES))
    model_with_noise = model + white_noise

    uncertainties_safe = np.maximum(uncertainties, 3e-6)
    residuals_diff = residuals_gw - model_with_noise
    chi2 = np.sum(data_weights * np.nansum((residuals_diff)**2 / uncertainties_safe**2, axis=1)) / (N_PULSARS * N_TIMES) * 1e2
    if gw_fitness.eval_count <= 5:
        logger.debug(f"Residuals diff max: {np.nanmax(np.abs(residuals_diff))}")
        logger.debug(f"Chi2 raw sum: {np.sum(data_weights * np.nansum((residuals_diff)**2 / uncertainties_safe**2, axis=1))}")

    gw_centered = gw_only - np.mean(gw_only, axis=1, keepdims=True)
    gw_std = np.std(gw_only, axis=1, keepdims=True) + 1e-12
    corr = np.dot(gw_centered, gw_centered.T) / (N_TIMES * gw_std * gw_std.T)
    corr_off_diag = corr[i_indices, j_indices]
    if gw_fitness.eval_count <= 5:
        logger.debug(f"Corr off-diag sample: {corr_off_diag[:5]}")

    hd_penalty = np.sum((corr_off_diag - hd_theoretical)**2) * 1e4

    earth_penalty = 0

    total_fitness = chi2 + hd_penalty + earth_penalty

    if gw_fitness.eval_count <= 5:
        logger.info(f"Fitness evaluation {gw_fitness.eval_count} took {time.time() - start:.2f}s")
    
    if return_full:
        return total_fitness, chi2, hd_penalty, earth_penalty, start
    return total_fitness

# ... (rest unchanged)


if __name__ == "__main__":
    data_dir = r'/home/wesley/data'