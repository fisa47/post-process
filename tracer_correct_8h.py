import xarray as xr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import dask
from dask.diagnostics import ProgressBar

def sanitize_string(s):
    """Replaces spaces and commas with underscores."""
    return s.replace(' ', '_').replace(',', '_')

# Constants
fvcom_origin = datetime(1858, 11, 17)
conv = 1e6/3.6  # from kg/hr to ug/s

# Load dataset
# Here it is a dataset with 2 hr timestamps merged from restart files
ds_all = xr.open_dataset('/Users/Admin/Documents/scripts/fvcom-work/Lysefjord/output/lysefjord_tracers_raw.nc',
                           decode_times=False)

ds_all = ds_all[['time', 'node', 'siglay', 'h', 'zeta', 'river_tracer_c', 'tracer2_c', 'tracer4_c', 'tracer6_c']]

ds_all = ds_all.chunk({'time': 24})
time = ds_all.time.values

# Create a copy of tracer4_c for Tralopyril
ds_all[f'tracer4_c_copy'] = ds_all['tracer4_c'] * 1


# Lastabotn:
# Mass flux = 0.00064 g/s -> 0.64 mg/s if we assume flux of 40 L/s -> 0.04 m3/s then concentration is 16 mg/m3

# Discharge = 10 L/s -> 0.01 m3/s
tracers = [
   {'name': 'river_tracer_c', 'title': 'Lastabotn Cu',
    'dummy_river_discharge': 10.0/60, 'dummy_river_conc': 10,   # DUMMY VALUES set in model
    'real_c': 16*1e3,  # ug/m3
    'real_discharge': 0.01,  # m3/s
    'release_start': time[0], 'loc': (-10115, 6566772)},

   {'name': 'tracer2_c', 'title': 'Ådnøy SØ 8 hours, Cu',
    'dummy_mass_flux': 70, 'real_mass_flux': 2*650/8*conv, # ug/s
    'release_start':  58344.166667,  # 2018-08-14 4:00:00
    'loc': (-15715, 6565218), 'pipe_ind': 37284},

   {'name': 'tracer4_c', 'title': 'Oltenvik 8 hours, Cu',
    'dummy_mass_flux': 90, 'real_mass_flux': 2*262/8*conv,
    'release_start': 58344.375000,  # 2018-08-14 9:00:00
    'loc': (-11046, 6557939), 'pipe_ind': 49112},

#    {'name': 'tracer4_c_copy', 'title': 'Oltenvik 8 hours, Tralapyril', 
#     'dummy_mass_flux': 90, 'real_mass_flux': 10.4/8*conv,
#     'release_start': 58344.375000,  # 2018-08-14 9:00:00
#     'loc': (-11046, 6557939), 'pipe_ind': 49112},

   {'name': 'tracer6_c', 'title': 'Gråtnes 8 hours',
    'dummy_mass_flux': 80, 'real_mass_flux': 2*356/8*conv,
    'release_start': 58344.375000,  # 2018-08-14 9:00:00
    'loc': (-10977, 6559806), 'pipe_ind': 48198},
]

def unstructured_grid_volume(area, depth, surface_elevation, thickness, depth_integrated=False):
    """
    Calculate the volume for every cell in the unstructured grid.

    Parameters
    ----------
    area : np.ndarray
        Element area
    depth : np.ndarray
        Static water depth
    surface_elevation : np.ndarray
        Time-varying surface elevation
    thickness : np.ndarray
        Level (i.e. between layer) position (range 0-1). In FVCOM, this is siglev.
    depth_intergrated : bool, optional
        Set to True to return the depth-integrated volume in addition to the depth-resolved volume. Defaults to False.

    Returns
    -------
    depth_volume : np.ndarray
        Depth-resolved volume of all the elements with time.
    volume : np.ndarray, optional
        Depth-integrated volume of all the elements with time.

    """

    # Convert thickness to actual thickness rather than position in water column of the layer.
    dz = np.abs(np.diff(thickness, axis=0))
    volume = (area * (surface_elevation + depth))
    depth_volume = volume[:, np.newaxis, :] * dz[np.newaxis, ...]

    if depth_integrated:
        return depth_volume, volume
    else:
        return depth_volume

def correct_tracer_concentration(
    ds,                      # xarray.Dataset containing tracer fields
    tracer_name,             # str, name of tracer variable in ds
    vol,                     # np.ndarray, grid cell volumes [shape: (time, layer, node)]
    time,                    # np.ndarray, FVCOM time values [shape: (time,)]
    dummy_river_conc=None,        # float, dummy concentration in µg/m³ (for river tracers)
    dummy_river_discharge=None,     # float, dummy discharge in m³/s (for river tracers)
    real_river_conc=None, # float, real concentration in µg/m³ (for river tracers)
    real_river_discharge=None,     # float, real discharge in m³/s (for river tracers)
    dummy_mass_flux=None,      # float, dummy mass flux in µg/s (for injection tracers)
    real_mass_flux=None, # float, real mass flux in µg/s (for injection tracers)
    release_start_time=None,      # float, FVCOM time value when release starts
    release_duration_seconds=8*3600 # float, duration of pulse release in seconds (default 8 hours),
):
    """
    Correct tracer concentration in the dataset.
    Returns tracer concentration in µg/L. Automatically converts from µg/m³ to µg/L.
    """
    # For mass flux injection
    use_inject_mode = dummy_mass_flux is not None and real_mass_flux is not None
    # For rivers (given discharge and concentration)
    use_river_mode = dummy_river_conc is not None and dummy_river_discharge is not None and real_river_conc is not None

    if not (use_inject_mode or use_river_mode):
        raise ValueError("Provide either (dummy_river_conc + dummy_river_discharge + real_river_conc) "
                         "OR (dummy_mass_flux + real_mass_flux)")

    if release_start_time is None:
        raise ValueError("You must provide `release_start_time` for each tracer.")

    # Time (in seconds) since release started
    seconds_since_release = np.maximum((time - release_start_time) * 86400, 0)

    if use_inject_mode:
        
        release_end_time = release_start_time + release_duration_seconds/86400
        iend = np.argwhere(time <= release_end_time)[-1][0]
        #print("iend:", iend)
        
        # Identify time steps during the release period (release_start_time to release_start_time + release_duration_seconds)
        during_release = ((time >= release_start_time) & (time <= release_end_time))
        after_release = time > release_end_time

        # Initialize array of zeros to hold the cumulative mass discharged at each time step
        # Before release mass discharged is 0
        target_mass = np.zeros_like(time)

        # For times during the release, calculate cumulative mass as real_mass_flux * elapsed seconds since release started
        target_mass[during_release] = real_mass_flux * (seconds_since_release[during_release])

        # For times after the release period, total mass released is mass_flux * release_duration_seconds (full pulse delivered)
        target_mass[after_release] = real_mass_flux  * release_duration_seconds # constant value

        # Compute model dummy mass discharge to compare with theoretical
        dummy_mass_model = (vol * ds[tracer_name]).sum(dim=('node', 'siglay')).values
        plt.figure(figsize=(8,4))
        plt.plot(dummy_mass_model)
        plt.title('dummy mass model')
        plt.savefig('dummy_model_raw.png', dpi=200)
        mass_dummy_final = dummy_mass_model[iend]
        dummy_mass_model[after_release] = mass_dummy_final

    else:
        if real_river_discharge is None:
            raise ValueError("You must provide `real_river_discharge` for river tracers.")
        target_mass = real_river_conc * real_river_discharge * seconds_since_release

        # Compute integrated dummy tracer mass at each timestep (sum over space)
        dummy_mass_model = (vol * ds[tracer_name]).sum(dim=('node', 'siglay'))  # should be xarray


    # Allocate output array (same shape as dummy tracer)
    tracer_shape = ds[tracer_name].shape
    tracer_conc = np.empty(tracer_shape, dtype=np.float32)

    n_time = ds[tracer_name].shape[0]
    tracer_conc = np.empty_like(ds[tracer_name].values, dtype=np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        correction_factor2 = np.where(dummy_mass_model > 0, target_mass / dummy_mass_model, 1.0)
        correction_factor2[0] = 1.0
        if use_inject_mode:
            # constant correction from the moment when release stopped
            target_mass_final = target_mass[iend]
            dummy_mass_final = dummy_mass_model[iend]
            correction_factor2 = target_mass_final / dummy_mass_final
            print('correction 2: ', correction_factor2)

    # second correction
    real_mass_corrected = np.empty_like(vol)
    for t in range(n_time):
        field = ds[tracer_name][t].values           # (layer, node)
        volume_t = vol[t]                           # (layer, node)
        # Compute real corrected mass
        if use_inject_mode:
            real_mass_corrected[t] = field * volume_t * correction_factor2
        else:
            real_mass_corrected[t] = field * volume_t * correction_factor2[t]

        # correct concentrations
        tracer_conc[t] = real_mass_corrected[t] / volume_t / 1000  # µg/L

    #print("real_mass_corrected, kg: ", real_mass_corrected.sum(axis=(1,2)) / 1e9)
     
    ### plot
    plt.figure(figsize=(8, 4))
    plt.plot(time, dummy_mass_model, label='model dummy mass')
    plt.legend()
    plt.savefig(f"dummy_mass_{tracer_name}.png")
    
    # Reporting
    print(f"Tracer '{tracer_name}' corrected using:")
    if use_inject_mode:
        print(f"  dummy_flux     = {dummy_mass_flux:.2e} µg/s")
        print(f"  real_flux      = {real_mass_flux:.2e} µg/s")
    else:
        print(f"  dummy_conc     = {dummy_river_conc} µg/L")
        print(f"  dummy_discharge= {dummy_river_discharge} m³/s")
        if real_river_discharge is not None:
            print(f"  real_discharge = {real_river_discharge} m³/s")
        print(f"  real_conc      = {real_river_conc} µg/m³")
    print(f"  release_start  = {release_start_time}")
    print(f"  release_dur    = {release_duration_seconds / 3600:.1f} h")

    return tracer_conc


# Grid data
# Get it from actual output file
ds_output = xr.open_dataset('/Users/Admin/Documents/scripts/fvcom-work/Lysefjord/output/jul14/lysefjord_0001.nc', decode_times=False).isel(time=-1)
area = ds_output['art1'].values
depth = ds_output['h'].values
surface_elevation = ds_all['zeta'].values
thickness = ds_output['siglev'].values

tri = ds_output['nv'].values.T - 1
vol = unstructured_grid_volume(area, depth, surface_elevation, thickness)
x, y = ds_output['x'], ds_output['y']


for tracer in tracers:
    tracer_name = tracer['name']

    # --- Run correction ---
    if 'dummy_mass_flux' in tracer:
        conc = correct_tracer_concentration(
            ds_all, tracer_name, vol, time,
            dummy_mass_flux=tracer['dummy_mass_flux'],
            real_mass_flux=tracer['real_mass_flux'],
            release_start_time=tracer['release_start']
        )
    else:
        conc = correct_tracer_concentration(
            ds_all, tracer_name, vol, time,
            dummy_river_conc=tracer['dummy_river_conc'],
            dummy_river_discharge=tracer['dummy_river_discharge'],
            real_river_conc=tracer['real_c'],
            real_river_discharge=tracer['real_discharge'],
            release_start_time=tracer['release_start'],
        )

    # Store
    ds_all[f'{tracer_name}_corrected'] = ds_all[tracer_name] * 0
    ds_all[f'{tracer_name}_corrected'].values = conc


ds_all = ds_all[['time', 'node', 'siglay', 'h', 'zeta',
        'river_tracer_c_corrected', 'tracer2_c_corrected',
        'tracer4_c_corrected', 'tracer6_c_corrected']]

delayed_obj = ds_all.to_netcdf('/Users/Admin/Documents/scripts/fvcom-work/Lysefjord/output/lysefjord_tracers_corrected_2h_M2.nc',
                                compute=False)
with ProgressBar():
    dask.compute(delayed_obj)
