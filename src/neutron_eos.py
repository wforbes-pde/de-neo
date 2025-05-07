import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.stats import gaussian_kde
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Constants (in cgs units)
G = 6.67430e-8
c = 2.99792458e10
Msun = 1.98847e33
km_to_cm = 1e5
R0 = 1e6  # 10 km
M0 = Msun
P0 = 1e34
rho0 = 1e15

# Logging control
density_log_count = 0
max_density_logs = 1
zero_pressure_logged = False
invalid_deriv_logged = False
tov_step_count = 0
max_tov_steps = 5
generation = 0
obj_eval_count = 0
solver_step_count = 0
max_solver_steps = 50
solver_terminated = False

# Read data
try:
    with h5py.File('GW170817_GWTC-1.hdf5', 'r') as f:
        dataset = f['IMRPhenomPv2NRT_lowSpin_posterior']
        if 'lambda1' not in dataset.dtype.names or 'lambda2' not in dataset.dtype.names:
            raise KeyError(f"lambda1 or lambda2 not found. Available fields: {dataset.dtype.names}")
        lambda1 = dataset['lambda1'][:]
        lambda2 = dataset['lambda2'][:]
except KeyError as e:
    print(f"KeyError: {e}. Inspect HDF5 structure with:")
    print("""
    with h5py.File('GW170817_GWTC-1.hdf5', 'r') as f:
        print("Top-level keys:", list(f.keys()))
        dataset = f['IMRPhenomPv2NRT_lowSpin_posterior']
        print("Dataset fields:", dataset.dtype.names)
    """)
    raise

lambda_samples = np.concatenate([lambda1, lambda2])
lambda_kde = gaussian_kde(lambda_samples)

try:
    data_j0030 = np.loadtxt('J0030_3spot_RM.txt', skiprows=4)
    radius1 = data_j0030[:, 0]
    mass1 = data_j0030[:, 1]
    print(f"PSR J0030+0451: Mass range {mass1.min():.2f}-{mass1.max():.2f} Msun, "
          f"Radius range {radius1.min():.2f}-{radius1.max():.2f} km")
except (IndexError, ValueError) as e:
    print(f"Error reading J0030_3spot_RM.txt: {e}. File contents:")
    with open('J0030_3spot_RM.txt', 'r') as f:
        print(f.read()[:500])
    raise

mass1_kde = gaussian_kde(mass1)
radius1_kde = gaussian_kde(radius1)

try:
    data_j0740 = np.loadtxt('NICER+XMM_J0740_RM.txt', skiprows=4)
    radius2 = data_j0740[:, 0]
    mass2 = data_j0740[:, 1]
    weights2 = data_j0740[:, 2]
    print(f"PSR J0740+6620: Mass range {mass2.min():.2f}-{mass2.max():.2f} Msun, "
          f"Radius range {radius2.min():.2f}-{radius2.max():.2f} km")
except (IndexError, ValueError) as e:
    print(f"Error reading NICER+XMM_J0740_RM.txt: {e}. File contents:")
    with open('NICER+XMM_J0740_RM.txt', 'r') as f:
        print(f.read()[:500])
    raise

mass2_kde = gaussian_kde(mass2, weights=weights2)
radius2_kde = gaussian_kde(radius2, weights=weights2)

# Piecewise polytropic EOS
def piecewise_polytropic_eos(rho, params):
    n_segments = 4
    K = params[:n_segments].copy()
    gamma = params[n_segments:2*n_segments]
    rho_bounds = params[2*n_segments:2*n_segments + n_segments - 1]
    crust_K, crust_gamma = params[-4], params[-3]
    high_density_factor, max_mass_factor = params[-2], params[-1]
    
    rho_bounds = np.sort(rho_bounds)
    min_spacing = 1e14
    for i in range(1, len(rho_bounds)):
        if rho_bounds[i] - rho_bounds[i-1] < min_spacing:
            rho_bounds[i] = rho_bounds[i-1] + min_spacing
    rho_bounds = np.concatenate(([1e14], rho_bounds, [1e16]))
    
    if rho < 1e14:
        P = crust_K * rho**crust_gamma
        return max(P, 1e-10)
    
    P_prev = crust_K * 1e14**crust_gamma
    K[0] = P_prev / (1e14**gamma[0] * high_density_factor)
    P_segments = [P_prev]
    for i in range(1, n_segments):
        P_prev = K[i-1] * rho_bounds[i]**gamma[i-1] * high_density_factor
        K[i] = P_prev / (rho_bounds[i]**gamma[i] * high_density_factor)
        P_segments.append(P_prev)
        if K[i] <= 0 or not np.isfinite(K[i]) or P_segments[i] < P_segments[i-1]:
            return 1e-10
    
    blend_width = 4e14
    for i in range(n_segments):
        if rho_bounds[i] <= rho < rho_bounds[i+1]:
            if i < n_segments - 1 and rho > rho_bounds[i+1] - blend_width:
                frac = (rho - (rho_bounds[i+1] - blend_width)) / blend_width
                P1 = K[i] * rho**gamma[i] * high_density_factor
                P2 = K[i+1] * rho**gamma[i+1] * high_density_factor
                P = (1 - frac) * P1 + frac * P2
            else:
                P = K[i] * rho**gamma[i] * high_density_factor
            return max(P, 1e-10)
    
    P = K[-1] * rho**gamma[-1] * high_density_factor
    return max(P, 1e-10)

def density_from_pressure(P, params):
    global density_log_count
    n_segments = 4
    K = params[:n_segments].copy()
    gamma = params[n_segments:2*n_segments]
    rho_bounds = params[2*n_segments:2*n_segments + n_segments - 1]
    crust_K, crust_gamma = params[-4], params[-3]
    high_density_factor, max_mass_factor = params[-2], params[-1]
    
    rho_bounds = np.sort(rho_bounds)
    min_spacing = 1e14
    for i in range(1, len(rho_bounds)):
        if rho_bounds[i] - rho_bounds[i-1] < min_spacing:
            rho_bounds[i] = rho_bounds[i-1] + min_spacing
    rho_bounds = np.concatenate(([1e14], rho_bounds, [1e16]))
    
    log_density = density_log_count < max_density_logs
    if log_density:
        print(f"Density calc: P={P:.2e}, K={K.tolist()}, high_density_factor={high_density_factor:.2f}")
        density_log_count += 1
    
    if P <= 1e-10:
        return 1e-10
    
    P_crust = crust_K * 1e14**crust_gamma
    K[0] = P_crust / (1e14**gamma[0] * high_density_factor)
    P_segments = [P_crust]
    for i in range(1, n_segments):
        P_prev = K[i-1] * rho_bounds[i]**gamma[i-1] * high_density_factor
        K[i] = P_prev / (rho_bounds[i]**gamma[i] * high_density_factor)
        P_segments.append(P_prev)
        if K[i] <= 0 or not np.isfinite(K[i]) or P_segments[i] < P_segments[i-1]:
            return 1e-10
    
    if P < P_crust:
        rho = (P / crust_K)**(1.0 / crust_gamma)
        if not np.isfinite(rho) or rho <= 0:
            return 1e-10
        return min(rho, 2e15)
    
    blend_width = 4e14
    for i in range(n_segments):
        rho_low, rho_high = rho_bounds[i], rho_bounds[i+1]
        P_low = K[i] * rho_low**gamma[i] * high_density_factor
        P_high = K[i] * rho_high**gamma[i] * high_density_factor
        if i < n_segments - 1 and P >= P_high - 1e12:
            P_next = K[i+1] * rho_high**gamma[i+1] * high_density_factor
            if P_high <= P < P_next:
                frac = (P - P_high) / (P_next - P_high)
                rho = rho_high + frac * (rho_bounds[i+2] - rho_high)
                if not np.isfinite(rho) or rho <= 0:
                    return 1e-10
                return min(rho, 2e15)
        if P_low <= P < P_high:
            rho = (P / (K[i] * high_density_factor))**(1.0 / gamma[i])
            if not np.isfinite(rho) or rho <= 0:
                return 1e-10
            return min(rho, 2e15)
    
    return 2e15

# Check EOS monotonicity
def check_eos_monotonicity(params):
    n_segments = 4
    rho_test = np.logspace(14, 16, 100)
    P_prev = 0
    for i, rho in enumerate(rho_test):
        P = piecewise_polytropic_eos(rho, params)
        if P < P_prev or not np.isfinite(P):
            return False
        if i > 0:
            dP = (P - P_prev) / (rho - rho_test[i-1])
            if dP < 0 or not np.isfinite(dP):
                return False
        P_prev = P
    
    rho_bounds = params[2*n_segments:2*n_segments + n_segments - 1]
    if len(rho_bounds) != n_segments - 1:
        return False
    rho_bounds = np.sort(rho_bounds)
    min_spacing = 1e14
    for i in range(1, len(rho_bounds)):
        if rho_bounds[i] - rho_bounds[i-1] < min_spacing:
            rho_bounds[i] = rho_bounds[i-1] + min_spacing
    rho_bounds = np.concatenate(([1e14], rho_bounds, [1e16]))
    boundary_pressures = [piecewise_polytropic_eos(rho, params) for rho in rho_bounds]
    print(f"Boundary check: rho={rho_bounds}, P={boundary_pressures}")
    return True

# TOV equations (non-relativistic)
def tov_equations(r_scaled, state, params):
    global zero_pressure_logged, invalid_deriv_logged, tov_step_count, solver_step_count, solver_terminated
    if solver_terminated:
        return [0, 0]
    solver_step_count += 1
    if solver_step_count > max_solver_steps:
        print(f"Excessive solver steps: {solver_step_count}")
        solver_terminated = True
        return [0, 0]
    P_scaled, m_scaled = state
    r = r_scaled * R0
    P = P_scaled * P0
    m = m_scaled * M0
    if P <= 1e-4:
        if not zero_pressure_logged:
            print(f"Zero pressure at r={r/km_to_cm:.2f} km")
            zero_pressure_logged = True
            solver_terminated = True
        return [0, 0]
    if P > 1e40 or m > 10 * Msun:
        if not invalid_deriv_logged:
            print(f"Overflow: r={r/km_to_cm:.2f} km, P={P:.2e}, m={m/Msun:.2f} Msun")
            invalid_deriv_logged = True
            solver_terminated = True
        return [0, 0]
    rho = density_from_pressure(P, params)
    if rho <= 0 or not np.isfinite(rho):
        if not invalid_deriv_logged:
            print(f"Unphysical density: r={r/km_to_cm:.2f} km, P={P:.2e}, rho={rho:.2e}")
            invalid_deriv_logged = True
            solver_terminated = True
        return [0, 0]
    if rho > 2e15:
        if not invalid_deriv_logged:
            print(f"Excessive density: r={r/km_to_cm:.2f} km, P={P:.2e}, rho={rho:.2e}")
            invalid_deriv_logged = True
            solver_terminated = True
        return [0, 0]
    rho_scaled = rho / rho0
    r_scaled = r / R0
    m_scaled = m / M0
    dP_dr_scaled = - (G * M0 / R0**2) * rho_scaled * m_scaled / (r_scaled**2 + 1e-6) * 1e-12
    dm_dr_scaled = 4 * np.pi * r_scaled**2 * rho_scaled * (rho0 * R0**3 / M0)
    if not np.isfinite(dP_dr_scaled) or not np.isfinite(dm_dr_scaled):
        if not invalid_deriv_logged:
            print(f"Invalid derivatives: r={r/km_to_cm:.2f} km, P={P:.2e}, m={m/Msun:.2f} Msun, rho={rho:.2e}")
            invalid_deriv_logged = True
            solver_terminated = True
        return [0, 0]
    if r/km_to_cm >= 0.01 and r <= 3e6 and tov_step_count < max_tov_steps and solver_step_count <= max_solver_steps:
        print(f"TOV step: r={r/km_to_cm:.2f} km, P={P:.2e}, rho={rho:.2e}, m={m/Msun:.2f} Msun, dP_dr_scaled={dP_dr_scaled:.2e}")
        tov_step_count += 1
    return [dP_dr_scaled, dm_dr_scaled]

# Solve TOV equations
def solve_tov(params, P_central=1e34, r_max=3e6, n_points=2000):
    global zero_pressure_logged, invalid_deriv_logged, density_log_count, tov_step_count, solver_step_count, solver_terminated
    zero_pressure_logged = False
    invalid_deriv_logged = False
    density_log_count = 0
    tov_step_count = 0
    solver_step_count = 0
    solver_terminated = False
    r_span = (5000.0 / R0, r_max / R0)
    state0 = [P_central / P0, 0]
    
    def event(r_scaled, state, params):
        return state[0] * P0 - 1e-4
    event.terminal = True
    event.direction = -1
    
    print(f"Starting TOV: P_central={P_central:.2e}")
    try:
        sol = solve_ivp(
            tov_equations,
            r_span,
            state0,
            args=(params,),
            method='RK45',
            rtol=1e-13,
            atol=1e-15,
            max_step=5e-1 / R0,
            events=event,
            dense_output=False
        )
    except Exception as e:
        print(f"TOV solver error: {e}")
        return 0.0, 0.0
    
    print(f"TOV result: status={sol.status}, message={sol.message}")
    if len(sol.t_events[0]) > 0:
        # Event triggered
        r_event = sol.t_events[0][-1] * R0
        idx = np.argmin(np.abs(sol.t * R0 - r_event))
        P = sol.y[0][idx] * P0
        m = sol.y[1][idx] * M0
        R = r_event / km_to_cm
        M = m / Msun
        print(f"TOV event: M={M:.2f} Msun, R={R:.2f} km")
        return M, R
    if not sol.success:
        print(f"TOV integration failed: {sol.message}, P_central={P_central:.2e}")
        print(f"Solver steps: {len(sol.t)}, r={sol.t*R0/km_to_cm}, P={sol.y[0]*P0}")
        return 0.0, 0.0
    
    r = sol.t * R0
    P_scaled, m_scaled = sol.y
    P = P_scaled * P0
    m = m_scaled * M0
    idx = np.where(P > 1e-4)[0][-1] if np.any(P > 1e-4) else -1
    R = r[idx] / km_to_cm if idx >= 0 else 0.0
    M = m[idx] / Msun if idx >= 0 else 0.0
    print(f"TOV completed: M={M:.2f} Msun, R={R:.2f} km")
    return M, R

# Tidal deformability (simplified)
def tidal_deformability(M, R, params):
    if M <= 0 or R <= 0:
        return 0.0
    compactness = G * M * Msun / (R * km_to_cm * c**2)
    Lambda = 1000 * (1 - compactness)**2
    return Lambda

# Observational data
data = {
    'stars': [
        {'M': 1.4, 'R': 11.9, 'Lambda': 300},
        {'M': 1.44, 'R': 12.7},
    ],
    'M_max': 2.0,
    'M_max_err': 0.1
}

# Objective function
def objective_function(params, data):
    global obj_eval_count, density_log_count
    obj_eval_count += 1
    density_log_count = 0
    try:
        if not check_eos_monotonicity(params):
            print(f"Eval {obj_eval_count}: Non-monotonic EOS")
            return 1e10
        
        chi2 = 0
        central_pressures = [1e25, 5e25]
        for i, star in enumerate(data['stars']):
            M, R = solve_tov(params, P_central=central_pressures[i])
            if M <= 0 or R <= 0 or R > 50:
                print(f"Eval {obj_eval_count}: TOV failed: M={M:.2f}, R={R:.2f}, P_central={central_pressures[i]:.2e}")
                return 1e8
            if 'Lambda' in star:
                Lambda = tidal_deformability(M, R, params)
                chi2 -= np.log(max(lambda_kde(Lambda), 1e-10))
            chi2 -= np.log(max(radius1_kde(R) if i == 1 else radius2_kde(R), 1e-10))
            chi2 -= np.log(max(mass1_kde(M) if i == 1 else mass2_kde(M), 1e-10))
        
        P_central_max = np.logspace(25, 26, 2)
        M_max = 0
        for P_c in P_central_max:
            M, R = solve_tov(params, P_central=P_c)
            if M > M_max:
                M_max = M
        chi2 += ((M_max - data['M_max']) / data['M_max_err'])**2
        
        rho_test = np.logspace(14, 16, 100)
        P_prev = 0
        for i, rho in enumerate(rho_test):
            P = piecewise_polytropic_eos(rho, params)
            if P / rho > c**2:
                chi2 += 1e5
            if P_prev > 0:
                dP = (P - P_prev) / (rho - rho_test[i-1])
                if dP < 0 or dP > 1e33:
                    chi2 += 1e5
            P_prev = P
        
        n_segments = 4
        rho_bounds = params[2*n_segments:2*n_segments + n_segments - 1]
        rho_bounds = np.sort(rho_bounds)
        min_spacing = 1e14
        for i in range(1, len(rho_bounds)):
            if rho_bounds[i] - rho_bounds[i-1] < min_spacing:
                rho_bounds[i] = rho_bounds[i-1] + min_spacing
        rho_bounds = np.concatenate(([1e14], rho_bounds, [1e16]))
        for i in range(len(rho_bounds) - 1):
            P1 = piecewise_polytropic_eos(rho_bounds[i], params)
            P2 = piecewise_polytropic_eos(rho_bounds[i] + 1e12, params)
            if abs(P2 - P1) / P1 > 0.02:
                chi2 += 1e6
            if P1 > 1e16:
                chi2 += 1e5 * (P1 / 1e16)
            if P1 < 1e32 and rho_bounds[i] > 5e14:
                chi2 += 1e6 * (1e32 / P1)
        print(f"Eval {obj_eval_count}: Objective chi2: {chi2:.2e}")
        return chi2
    except Exception as e:
        print(f"Eval {obj_eval_count}: Objective function error: {e}")
        return 1e10

# Parameter bounds
n_segments = 4
bounds = (
    [(50.0, 200.0)] * n_segments +  # Higher K
    [(1.98, 2.02)] * n_segments +
    [(5e15, 8e15)] * (n_segments - 1) +
    [(1e-14, 1e-12), (1.98, 2.02)] +
    [(3.0, 10.0), (0.95, 1.05)]
)

# DE callback
def de_callback(xk, convergence):
    global generation
    generation += 1
    chi2 = objective_function(xk, data)
    print(f"Generation {generation}: chi2={chi2:.2e}, K={xk[:4].tolist()}, high_density_factor={xk[-2]:.2f}")

# Run Differential Evolution
result = differential_evolution(
    objective_function,
    bounds,
    args=(data,),
    strategy='best1bin',
    maxiter=10,
    popsize=15,
    tol=1e-6,
    seed=42,
    disp=True,
    callback=de_callback,
    updating='deferred',
    workers=1
)

# Print results
print(f"Best-fit parameters: {result.x}")
print(f"Objective function value: {result.fun:.2f}")

# Plot and save mass-radius curve
central_pressures = np.logspace(25, 26, 10)
M_list, R_list = [], []
for P_c in central_pressures:
    M, R = solve_tov(result.x, P_central=P_c)
    if M > 0 and R > 0:
        M_list.append(M)
        R_list.append(R)

plt.figure()
plt.plot(R_list, M_list, label='Best-fit EOS')
for star in data['stars']:
    plt.plot(star['R'], star['M'], 'o', label=f'M={star["M"]} Msun')
plt.xlabel('Radius (km)')
plt.ylabel('Mass (Msun)')
plt.legend()
plt.savefig('mass_radius_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Mass-radius curve saved as 'mass_radius_curve.png'")