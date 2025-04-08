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


##################################

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



if __name__ == "__main__":
    #data_dir = "/home/wesley/data/narrowband"
    data_dir = r'/home/wesley/data'
    #times, residuals, uncertainties, positions = load_pulsar_data(data_dir, n_pulsars=20, fit_toas=False)