import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter
from pint.residuals import Residuals  # Move import here
from pint.models.glitch import Glitch
from pint.models.parameter import MJDParameter, floatParameter
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
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Suppress PINT logging to avoid clutter
pint.logging.setup(level="WARNING")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def fit_glitches(psr_name, model, toas, glitch_epochs):
    try:
        logging.info(f"Fitting glitches for {psr_name} at epochs: {glitch_epochs}")
        for i, epoch in enumerate(glitch_epochs, 1):
            glitch = Glitch()
            glitch_idx = f"_{i}"
            # GLEP: Add as MJDParameter (typically not pre-existing)
            glep_param = MJDParameter(
                name=f"GLEP{glitch_idx}",
                value=float(epoch),
                units="day",
                description=f"Epoch of glitch {i}",
                frozen=False,
                tcb2tdb_scale_factor=1.0
            )
            glep_param.index = i
            glitch.add_param(glep_param)
            # GLF0: Check if exists, update or add
            glf0_name = f"GLF0{glitch_idx}"
            if glf0_name in glitch.params:
                logging.info(f"{psr_name}: Updating existing {glf0_name}")
                glitch.GLF0_1.value = 0.0
                glitch.GLF0_1.frozen = False
                glitch.GLF0_1.index = i
            else:
                glf0_param = floatParameter(
                    name=glf0_name,
                    value=0.0,
                    units="Hz",
                    description=f"Frequency offset for glitch {i}",
                    frozen=False,
                    tcb2tdb_scale_factor=1.0
                )
                glf0_param.index = i
                glitch.add_param(glf0_param)
            # GLF1: Check if exists, update or add
            glf1_name = f"GLF1{glitch_idx}"
            if glf1_name in glitch.params:
                logging.info(f"{psr_name}: Updating existing {glf1_name}")
                glitch.GLF1_1.value = 0.0
                glitch.GLF1_1.frozen = False
                glitch.GLF1_1.index = i
            else:
                glf1_param = floatParameter(
                    name=glf1_name,
                    value=0.0,
                    units="Hz/s",
                    description=f"Frequency derivative for glitch {i}",
                    frozen=False,
                    tcb2tdb_scale_factor=1.0
                )
                glf1_param.index = i
                glitch.add_param(glf1_param)
            # GLF0D: Check if exists, update or add
            glf0d_name = f"GLF0D{glitch_idx}"
            if glf0d_name in glitch.params:
                logging.info(f"{psr_name}: Updating existing {glf0d_name}")
                glitch.GLF0D_1.value = 0.0
                glitch.GLF0D_1.frozen = False
                glitch.GLF0D_1.index = i
            else:
                glf0d_param = floatParameter(
                    name=glf0d_name,
                    value=0.0,
                    units="Hz",
                    description=f"Decay term for glitch {i}",
                    frozen=False,
                    tcb2tdb_scale_factor=1.0
                )
                glf0d_param.index = i
                glitch.add_param(glf0d_param)
            model.add_component(glitch)
            logging.info(f"Added Glitch component {i} for {psr_name} with GLEP{glitch_idx}={epoch}")
        model.validate()
        fitter = WLSFitter(toas=toas, model=model)
        fitter.fit_toas()
        logging.info(f"Completed fitting glitches for {psr_name}")
        return model, fitter
    except Exception as e:
        logging.error(f"Error fitting glitches for {psr_name}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise



def detect_glitches(times, residuals, sigma=5, min_separation=100):
    glitch_epochs = []
    for i in range(len(residuals)):
        try:
            if len(times[i]) != len(residuals[i]):
                logger.error(f"Pulsar {i+1}: Length mismatch: times={len(times[i])}, residuals={len(residuals[i])}")
                glitch_epochs.append(np.array([]))
                continue

            logger.info(f"Pulsar {i+1}: Input times shape={times[i].shape}, dtype={times[i].dtype}, any NaN={np.any(np.isnan(times[i]))}, any inf={np.any(np.isinf(times[i]))}")
            logger.info(f"Pulsar {i+1}: Input residuals shape={residuals[i].shape}, dtype={residuals[i].dtype}, any NaN={np.any(np.isnan(residuals[i]))}, any inf={np.any(np.isinf(residuals[i]))}")

            times_valid = ~np.isnan(times[i]) & ~np.isinf(times[i])
            residuals_valid = ~np.isnan(residuals[i]) & ~np.isinf(residuals[i])
            logger.info(f"Pulsar {i+1}: Times valid shape={times_valid.shape}, dtype={times_valid.dtype}, valid entries={np.sum(times_valid)}")
            logger.info(f"Pulsar {i+1}: Residuals valid shape={residuals_valid.shape}, dtype={residuals_valid.dtype}, valid entries={np.sum(residuals_valid)}")

            mask = times_valid & residuals_valid
            logger.info(f"Pulsar {i+1}: Combined mask shape={mask.shape}, dtype={mask.dtype}, valid entries={np.sum(mask)}, expected length={len(times[i])}")
            
            if np.sum(mask) < len(times[i]):
                invalid_indices = np.where(~mask)[0]
                logger.info(f"Pulsar {i+1}: Invalid entries at indices {invalid_indices}")
                for idx in invalid_indices:
                    logger.info(f"Pulsar {i+1}: Index {idx}: time={times[i][idx]}, residual={residuals[i][idx]}")

            if len(mask) != len(times[i]):
                logger.error(f"Pulsar {i+1}: Mask length mismatch: mask={len(mask)}, times={len(times[i])}")
                glitch_epochs.append(np.array([]))
                continue

            t_i = times[i][mask] / 86400.0 + 51544.5
            r_i = residuals[i][mask].astype(np.float64)
            logger.info(f"Pulsar {i+1}: After masking, t_i shape={t_i.shape}, r_i shape={r_i.shape}")
            logger.info(f"Pulsar {i+1}: t_i min={np.min(t_i)}, max={np.max(t_i)}, any NaN={np.any(np.isnan(t_i))}, any inf={np.any(np.isinf(t_i))}")
            logger.info(f"Pulsar {i+1}: r_i min={np.min(r_i)}, max={np.max(r_i)}, any NaN={np.any(np.isnan(r_i))}, any inf={np.any(np.isinf(r_i))}")

            if len(t_i) < 2:
                logger.info(f"Pulsar {i+1}: Insufficient data points for glitch detection ({len(t_i)} points).")
                glitch_epochs.append(np.array([]))
                continue

            sort_idx = np.argsort(t_i)
            t_i = t_i[sort_idx]
            r_i = r_i[sort_idx]

            r_smooth = gaussian_filter1d(r_i, sigma=3)

            dt = np.diff(t_i)
            dr = np.diff(r_smooth)
            logger.info(f"Pulsar {i+1}: dt shape={dt.shape}, dr shape={dr.shape}")
            valid_dt = dt > 1e-6
            logger.info(f"Pulsar {i+1}: valid_dt shape={valid_dt.shape}, valid entries={np.sum(valid_dt)}")
            if not np.any(valid_dt):
                logger.info(f"Pulsar {i+1}: Time differences too small for glitch detection.")
                glitch_epochs.append(np.array([]))
                continue

            dr_dt = np.zeros_like(r_smooth)
            logger.info(f"Pulsar {i+1}: dr_dt shape={dr_dt.shape}, valid_dt shape={valid_dt.shape}, dr shape={dr.shape}, dt shape={dt.shape}")
            dr_dt[:-1][valid_dt] = dr[valid_dt] / dt[valid_dt]
            dr_dt[:-1][~valid_dt] = 0
            dr_dt[-1] = dr_dt[-2] if len(dr_dt) > 1 else 0

            # Compute standard deviation over valid derivatives
            dr_std = np.std(dr_dt[:-1][valid_dt]) if np.any(valid_dt) else 1.0
            logger.info(f"Pulsar {i+1}: dr_std={dr_std}")
            if dr_std == 0:
                logger.info(f"Pulsar {i+1}: No variation in derivatives for glitch detection.")
                glitch_epochs.append(np.array([]))
                continue

            peaks, properties = find_peaks(np.abs(dr_dt), height=sigma * dr_std, distance=min_separation, prominence=dr_std)
            glitch_mjds = t_i[peaks]
            if len(glitch_mjds) > 0:
                amplitudes = np.abs(r_smooth[peaks] - np.mean(r_smooth))
                significant = amplitudes > 3 * np.std(r_smooth)
                glitch_mjds = glitch_mjds[significant]

            glitch_epochs.append(glitch_mjds)
            logger.info(f"Pulsar {i+1}: Detected {len(glitch_mjds)} glitches at MJDs {glitch_mjds}")
        except Exception as e:
            logger.error(f"Pulsar {i+1}: Error in glitch detection: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            glitch_epochs.append(np.array([]))
    return glitch_epochs




# def fit_glitches(psr_name, model, toas, glitch_epochs):
#     if "Glitch" not in model.components:
#         model.add_component(Glitch())
#     if len(glitch_epochs) > 0:
#         for idx, epoch in enumerate(glitch_epochs, 1):
#             model[f'GLEP_{idx}'].quantity = epoch * u.day
#             model[f'GLPH_{idx}'].quantity = 0.0 * u.dimensionless_unscaled
#             model[f'GLF0_{idx}'].quantity = 0.0 * u.Hz
#             model[f'GLF1_{idx}'].quantity = 0.0 * u.Hz / u.s
#             model[f'GLF0_{idx}'].frozen = False
#             model[f'GLF1_{idx}'].frozen = False
#     fitter = WLSFitter(toas=toas, model=model)
#     fitter.fit_toas()
#     logger.info(f"Fitted glitches for {psr_name}:")
#     for idx in set(model.components["Glitch"].glitch_indices):
#         logger.info(f"Glitch {idx}: GLEP={model[f'GLEP_{idx}'].quantity}, "
#                     f"GLF0={model[f'GLF0_{idx}'].quantity}, "
#                     f"GLF1={model[f'GLF1_{idx}'].quantity}")
#     return model, fitter


def process_single_pulsar(args, fit_toas=True):
    psr_name, par_file, tim_file = args
    try:
        logging.info(f"Loading data for {psr_name} (par: {par_file}, tim: {tim_file})")
        model = get_model(par_file)
        #logging.info(f"{psr_name}: Model loaded successfully. Parameters: {[p for p in model.params]}")
        
        toas = get_TOAs(tim_file, model=model)
        logging.info(f"{psr_name}: TOAs loaded successfully. Number of TOAs: {len(toas)}")
        
        mjds = np.asarray(toas.get_mjds().value, dtype=np.float64).flatten()
        logging.info(f"{psr_name}: MJDs shape={mjds.shape}, dtype={mjds.dtype}, any NaN={np.any(np.isnan(mjds))}, any inf={np.any(np.isinf(mjds))}")
        logging.info(f"{psr_name}: Loaded {len(mjds)} TOAs, MJD range [{mjds.min():.2f}, {mjds.max():.2f}]")
        
        if np.any(np.diff(mjds) <= 0):
            logging.warning(f"Non-monotonic or duplicate MJDs detected for {psr_name}. Sorting TOAs.")
            toas.table.sort('mjd')
            mjds = np.asarray(toas.get_mjds().value, dtype=np.float64).flatten()
            logging.info(f"{psr_name}: After sorting, MJDs shape={mjds.shape}, dtype={mjds.dtype}")
        
        flags = toas.table['flags']
        unique_flags = set()
        if len(flags) > 0:
            unique_flags = set([f for flag_dict in flags for f in flag_dict.keys()])
        logging.info(f"{psr_name}: TOA flags: {unique_flags}")
        
        residuals_obj = Residuals(toas=toas, model=model)
        residuals = np.asarray(residuals_obj.time_resids.to("s").value, dtype=np.float64).flatten()
        logging.info(f"{psr_name}: Residuals shape={residuals.shape}, dtype={residuals.dtype}, any NaN={np.any(np.isnan(residuals))}, any inf={np.any(np.isinf(residuals))}")
        logging.info(f"{psr_name}: Residuals computed, length={len(residuals)}, min={np.min(residuals):.2e}, max={np.max(residuals):.2e}")
        
        if len(mjds) != len(residuals):
            logging.error(f"Length mismatch for {psr_name}: mjds={len(mjds)}, residuals={len(residuals)}")
            return None
        
        times = (mjds - mjds[0]) * 86400.0
        logging.info(f"{psr_name}: Times shape={times.shape}, dtype={times.dtype}, any NaN={np.any(np.isnan(times))}, any inf={np.any(np.isinf(times))}")
        logging.info(f"{psr_name}: Times computed, length={len(times)}, min={np.min(times):.2e}, max={np.max(times):.2e}")
        
        valid_mask = (~np.isnan(mjds) & ~np.isinf(mjds) & 
                      ~np.isnan(residuals) & ~np.isinf(residuals) & 
                      ~np.isnan(times) & ~np.isinf(times))
        logging.info(f"{psr_name}: Valid mask shape={valid_mask.shape}, dtype={valid_mask.dtype}, valid entries={np.sum(valid_mask)}")
        
        if not np.any(valid_mask):
            logging.error(f"No valid TOAs for {psr_name} after filtering NaNs and infs.")
            return None
        
        times = times[valid_mask]
        residuals = residuals[valid_mask]
        logging.info(f"{psr_name}: After filtering, times shape={times.shape}, residuals shape={residuals.shape}")
        
        if len(times) != len(residuals):
            logging.error(f"Post-filter length mismatch for {psr_name}: times={len(times)}, residuals={len(residuals)}")
            return None
        if len(times) < 2:
            logging.warning(f"Insufficient data points for {psr_name} after filtering ({len(times)} points). Skipping glitch detection.")
            glitch_epochs = np.array([])
        else:
            logging.info(f"{psr_name}: First 5 times={times[:5]}, First 5 residuals={residuals[:5]}")
            logging.info(f"{psr_name}: Times min={np.min(times)}, max={np.max(times)}, any NaN={np.any(np.isnan(times))}, any inf={np.any(np.isinf(times))}")
            logging.info(f"{psr_name}: Residuals min={np.min(residuals)}, max={np.max(residuals)}, any NaN={np.any(np.isnan(residuals))}, any inf={np.any(np.isinf(residuals))}")
            times_input = np.array([times])
            residuals_input = np.array([residuals])
            logging.info(f"{psr_name}: detect_glitches input shapes: times={times_input.shape}, residuals={residuals_input.shape}")
            glitch_epochs = detect_glitches(times_input, residuals_input, sigma=5, min_separation=100)[0]
            logging.info(f"{psr_name}: detect_glitches returned {len(glitch_epochs)} epochs: {glitch_epochs}")
        
        # Initialize fitter to ensure it's always defined
        fitter = None
        if len(glitch_epochs) > 0:
            min_mjd = mjds.min()
            max_mjd = mjds.max()
            valid_epochs = [epoch for epoch in glitch_epochs if min_mjd <= epoch <= max_mjd]
            logging.info(f"{psr_name}: Valid glitch epochs: {valid_epochs}")
            if not valid_epochs:
                logging.warning(f"No valid glitch epochs for {psr_name} within MJD range [{min_mjd}, {max_mjd}].")
                try:
                    fitter = WLSFitter(toas=toas, model=model)
                    if fit_toas:
                        fitter.fit_toas()
                        logging.info(f"{psr_name}: Fitted without glitches (no valid epochs).")
                    else:
                        residuals_obj = Residuals(toas=toas, model=model)
                        fitter.resids = residuals_obj
                        logging.info(f"{psr_name}: Computed residuals without fitting (no valid epochs).")
                except Exception as e:
                    logging.error(f"Error fitting without glitches for {psr_name}: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    return None
            else:
                try:
                    model, fitter = fit_glitches(psr_name, model, toas, valid_epochs)
                    logging.info(f"{psr_name}: Successfully fitted glitches.")
                except Exception as e:
                    logging.error(f"Error fitting glitches for {psr_name}: {e}")
                    import traceback
                    logging.error(f"Traceback: {traceback.format_exc()}")
                    return None
        else:
            logging.info(f"No glitches detected for {psr_name}, fitting without glitches.")
            try:
                fitter = WLSFitter(toas=toas, model=model)
                if fit_toas:
                    fitter.fit_toas()
                    logging.info(f"{psr_name}: Fitted without glitches.")
                else:
                    residuals_obj = Residuals(toas=toas, model=model)
                    fitter.resids = residuals_obj
                    logging.info(f"{psr_name}: Computed residuals without fitting.")
            except Exception as e:
                logging.error(f"Error fitting without glitches for {psr_name}: {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
                return None
        
        if fitter is None:
            logging.error(f"No fitter defined for {psr_name}. Cannot compute residuals.")
            return None

        mjds = np.asarray(toas.get_mjds().value, dtype=np.float64).flatten()
        mjds_sorted = np.sort(mjds)
        times_i = (mjds - mjds_sorted[0]) * 86400.0
        obs_span = (mjds_sorted[-1] - mjds_sorted[0]) * 86400.0
        residuals_i = np.asarray(fitter.resids.time_resids.to("s").value, dtype=np.float64).flatten()
        uncertainties_i = np.asarray(toas.get_errors().to("s").value, dtype=np.float64).flatten()
        
        if len(times_i) != len(residuals_i) or len(times_i) != len(uncertainties_i):
            logging.error(f"Final output length mismatch for {psr_name}: times={len(times_i)}, residuals={len(residuals_i)}, uncertainties={len(uncertainties_i)}")
            return None
        
        if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
            coord = SkyCoord(model.RAJ.quantity, model.DECJ.quantity, frame='icrs')
        elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
            coord = SkyCoord(model.ELONG.quantity, model.ELAT.quantity, frame='barycentrictrueecliptic').transform_to('icrs')
        else:
            logging.error(f"{psr_name}: Neither RAJ/DECJ nor ELONG/ELAT found in model parameters.")
            return None
        position = np.array([coord.ra.rad, coord.dec.rad])
        noise_level = np.std(residuals_i)
        
        return {
            "name": psr_name,
            "times": times_i,
            "residuals": residuals_i,
            "uncertainties": uncertainties_i,
            "position": position,
            "obs_span": obs_span,
            "noise_level": noise_level
        }
    except Exception as e:
        logging.error(f"Error loading data for {psr_name}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

    



def load_pulsar_data(data_dir, n_pulsars=20, use_wideband=False, n_jobs=None, fit_toas=True, cache_file="pulsar_data_cache.pkl"):
    if os.path.exists(cache_file):
        logging.info(f"Loading cached data from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            return cached_data["times"], cached_data["residuals"], cached_data["uncertainties"], cached_data["positions"]
        except Exception as e:
            logging.warning(f"Failed to load cache file {cache_file}: {e}. Regenerating data...")

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
    failed_count = n_pulsars - len(pulsar_data)
    if failed_count > 0:
        logging.warning(f"Failed to process {failed_count} out of {n_pulsars} pulsars.")
    if len(pulsar_data) < n_pulsars:
        logging.error(f"Successfully loaded only {len(pulsar_data)} pulsars, but {n_pulsars} are requested.")
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
    times = np.full((n_pulsars, max_n_times), np.nan)
    residuals = np.full((n_pulsars, max_n_times), np.nan)
    uncertainties = np.full((n_pulsars, max_n_times), np.nan)

    for i in range(n_pulsars):
        n_times_i = len(times_list[i])
        times[i, :n_times_i] = times_list[i]
        residuals[i, :n_times_i] = residuals_list[i]
        uncertainties[i, :n_times_i] = uncertainties_list[i]

    positions = np.array(positions)

    # Cache the results
    cache_data = {
        "times": times,
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