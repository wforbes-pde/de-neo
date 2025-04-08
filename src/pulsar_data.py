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

# Suppress PINT logging to avoid clutter
pint.logging.setup(level="WARNING")

# Set up logging for parallel processes
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_single_pulsar(args, fit_toas=True):
    """
    Process a single pulsar's data (load model, TOAs, fit, and extract data).

    Parameters:
    -----------
    args : tuple
        (psr_name, par_file, tim_file)
    fit_toas : bool
        If True, perform a fit with WLSFitter; if False, use pre-fit residuals.

    Returns:
    --------
    dict : Pulsar data dictionary or None if an error occurs.
    """
    psr_name, par_file, tim_file = args
    try:
        logging.info(f"Loading data for {psr_name} (par: {par_file}, tim: {tim_file})")
        # Load the timing model (.par file)
        model = get_model(par_file)

        # Load the TOAs (.tim file)
        toas = get_TOAs(tim_file, model=model)

        # Compute residuals
        if fit_toas:
            # Fit the model to the TOAs using WLSFitter
            fitter = WLSFitter(toas=toas, model=model)
            fitter.fit_toas()  # Perform a quick fit to compute residuals
        else:
            # Use pre-fit residuals
            from pint.residuals import Residuals
            fitter = WLSFitter(toas=toas, model=model)  # Still need fitter for model access
            residuals_obj = Residuals(toas=toas, model=model)
            fitter.resids = residuals_obj  # Assign pre-fit residuals

        # Debug: Print model components and astrometry parameters
        logging.info(f"Model components for {psr_name}: {fitter.model.components.keys()}")
        logging.info(f"Astrometry parameters: {fitter.model.get_params_of_type('astrometry')}")

        # Get observation times (MJD) and convert to seconds
        mjds = toas.get_mjds().value
        mjds_sorted = np.sort(mjds)
        times_i = (mjds - mjds_sorted[0]) * 86400.0
        obs_span = (mjds_sorted[-1] - mjds_sorted[0]) * 86400.0

        # Get residuals (in seconds)
        residuals_i = fitter.resids.time_resids.to("s").value

        # Get measurement uncertainties (in seconds)
        uncertainties_i = toas.get_errors().to("s").value

        # Get pulsar position (RA, Dec in radians)
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

        # Estimate noise level
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

def load_pulsar_data(data_dir, n_pulsars=20, use_wideband=False, n_jobs=None, fit_toas=True):
    """
    Load pulsar timing data from the NANOGrav 15-year dataset in parallel.

    Parameters:
    -----------
    data_dir : str
        Path to the root directory of the NANOGrav 15-year dataset.
    n_pulsars : int
        Number of pulsars to select (default: 20).
    use_wideband : bool
        If True, use wideband data; if False, use narrowband data (default: False).
    n_jobs : int, optional
        Number of parallel jobs. If None, uses cpu_count() - 1.
    fit_toas : bool
        If True, perform a fit with WLSFitter; if False, use pre-fit residuals.

    Returns:
    --------
    times : list of arrays [n_pulsars]
        Observation times for each pulsar (in seconds since the first observation).
    residuals : array [n_pulsars, max_n_times]
        Post-fit residuals (in seconds), padded with NaNs for irregular lengths.
    uncertainties : array [n_pulsars, max_n_times]
        Measurement uncertainties (in seconds), padded with NaNs.
    positions : array [n_pulsars, 2]
        Pulsar sky positions (RA, Dec in radians).
    """
    # Determine the data directory (narrowband or wideband)
    data_type = "wideband" if use_wideband else "narrowband"
    data_path = os.path.join(data_dir, data_type)

    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Directory {data_path} not found. Ensure the NANOGrav 15-year dataset is correctly structured.")

    # Define paths to par and tim directories
    par_path = os.path.join(data_path, "par")
    tim_path = os.path.join(data_path, "tim")

    # Check if the par and tim directories exist
    if not os.path.exists(par_path):
        raise FileNotFoundError(f"Par directory {par_path} not found.")
    if not os.path.exists(tim_path):
        raise FileNotFoundError(f"Tim directory {tim_path} not found.")

    # Find all .par and .tim files
    par_files = [f for f in os.listdir(par_path) if f.endswith(".par")]
    tim_files = [f for f in os.listdir(tim_path) if f.endswith(".tim")]

    # Debug: Print the found files
    print(f"Found {len(par_files)} .par files in {par_path}: {par_files[:5] if par_files else 'None'}")
    print(f"Found {len(tim_files)} .tim files in {tim_path}: {tim_files[:5] if tim_files else 'None'}")

    # Match .par and .tim files by base filename (without extension)
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

    # Set the number of parallel jobs
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Using {n_jobs} parallel jobs to process {n_pulsars} pulsars.")

    # Process pulsars in parallel
    process_func = partial(process_single_pulsar, fit_toas=fit_toas)
    with Pool(processes=n_jobs) as pool:
        pulsar_data = pool.map(process_func, pulsar_files[:n_pulsars])

    # Filter out None results (failed pulsars)
    pulsar_data = [p for p in pulsar_data if p is not None]

    if len(pulsar_data) < n_pulsars:
        raise ValueError(f"Successfully loaded only {len(pulsar_data)} pulsars, but {n_pulsars} are requested.")

    # Sort pulsars by observation span (descending) and noise level (ascending)
    pulsar_data.sort(key=lambda x: (-x["obs_span"], x["noise_level"]))

    # Extract data for the selected pulsars
    times = []
    residuals_list = []
    uncertainties_list = []
    positions = []

    for pulsar in pulsar_data[:n_pulsars]:
        times.append(pulsar["times"])
        residuals_list.append(pulsar["residuals"])
        uncertainties_list.append(pulsar["uncertainties"])
        positions.append(pulsar["position"])

    # Convert to arrays, padding with NaNs for irregular lengths
    max_n_times = max(len(t) for t in times)
    residuals = np.full((n_pulsars, max_n_times), np.nan)
    uncertainties = np.full((n_pulsars, max_n_times), np.nan)

    for i in range(n_pulsars):
        n_times_i = len(times[i])
        residuals[i, :n_times_i] = residuals_list[i]
        uncertainties[i, :n_times_i] = uncertainties_list[i]

    positions = np.array(positions)

    # Log the selected pulsars
    print("Selected pulsars:")
    for i, pulsar in enumerate(pulsar_data):
        print(f"Pulsar {i+1}: {pulsar['name']}, Obs Span: {pulsar['obs_span']/(365.25*86400):.2f} years, "
              f"Noise Level: {pulsar['noise_level']:.2e} s")

    return times, residuals, uncertainties, positions


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

def interpolate_residuals(times, residuals, uncertainties, n_times=500, t_span=15 * 365.25 * 86400):
    """
    Interpolate residuals onto a regular grid.
    times: List of arrays [n_pulsars], observation times in seconds.
    residuals: Array [n_pulsars, max_n_times], residuals in seconds.
    uncertainties: Array [n_pulsars, max_n_times], uncertainties in seconds.
    n_times: Number of time points in the regular grid.
    t_span: Total time span in seconds.
    Returns:
        regular_times: Array [n_times], regular time grid in seconds.
        interp_residuals: Array [n_pulsars, n_times], interpolated residuals.
        interp_uncertainties: Array [n_pulsars, n_times], interpolated uncertainties.
    """
    regular_times = np.linspace(0, t_span, n_times)
    interp_residuals = np.zeros((len(times), n_times))
    interp_uncertainties = np.zeros((len(times), n_times))

    for i in range(len(times)):
        mask = ~np.isnan(residuals[i])
        t_i = times[i][mask]
        r_i = residuals[i][mask]
        u_i = uncertainties[i][mask]
        # Linear interpolation
        interp_residuals[i] = np.interp(regular_times, t_i, r_i, left=np.nan, right=np.nan)
        interp_uncertainties[i] = np.interp(regular_times, t_i, u_i, left=np.nan, right=np.nan)

    return regular_times, interp_residuals, interp_uncertainties



if __name__ == "__main__":
    #data_dir = "/home/wesley/data/narrowband"
    data_dir = r'/home/wesley/data'
    times, residuals, uncertainties, positions = load_pulsar_data(data_dir, n_pulsars=20, fit_toas=False)