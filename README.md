# Lysefjord FVCOM Post-Processing Scripts

## Input Data

The main input file is a merged NetCDF file created from FVCOM 2 hours restart files. This was necessary because FABM did not write the required tracer variables to the standard output files.
The example is for 8 hours release.

## Tracers

Summary of the main tracers used in the simulations:

- **Lastabotn Cu (`river_tracer_c`)**  
    Copper released from the Lastabotn river.  
    - Discharge: 10 L/min (model river), 0.01 m³/s (real-world)  
    - Concentration: 10 µg/L (model river), 16 mg/m³ (real-world)  
      - Note: Mass flux = 0.00064 g/s (0.64 mg/s) assuming a discharge of 40 L/s (0.04 m³/s), ->_ concentration of Cu 16 mg/m³. 
    - Release start: beginning of the simulation 
    - Location UTM33: (-10115, 6566772)

- **Ådnøy SØ 8 hours, Cu (`tracer2_c`)**  
    Copper released at Ådnøy SØ over 8 hours.  
    - Dummy mass flux: 70 µg/s = 10 µg/s × 7 release levels
    - Real mass flux: 2 (doubled release) × 650 kg / 8 hr × conv µg/s 
    - Release start: Simulation time 58344.166667 (2018-08-14 04:00:00)  
    - Location UTM33: (-15715, 6565218)

- **Oltenvik 8 hours, Cu (`tracer4_c`)**  
    Copper released at Oltenvik over 8 hours.  
    - Dummy mass flux: 90 µg/s = 10 µg/s * 9 release levels
    - Real mass flux: 2 (doubled release) × 262 kg / 8 hr × conv µg/s  
    - Release start: Simulation time 58344.375000 (2018-08-14 09:00:00)  
    - Location UTM33: (-11046, 6557939)

- **Oltenvik 8 hours, Tralapyril (`tracer4_c_copy`)**  
    Tralapyril released at Oltenvik, same timing and location as copper.  
    - Dummy mass flux: 90 µg/s = 10 µg/s * 9 release levels 
    - Real mass flux: 10.4 kg / 8 hr × conv µg/s  
    - Release start: Simulation time 58344.375000  
    - Location UTM33: (-11046, 6557939)

- **Gråtnes 8 hours (`tracer6_c`)**  
    Substance released at Gråtnes over 8 hours.  
    - Dummy mass flux: 80 µg/s = 10 µg/s * 8 release levels 
    - Real mass flux: 2 (doubled release) × 356 kg / 8 hr × conv µg/s 
    - Release start: Simulation time 58344.375000 (2018-08-14 09:00:00)  
    - Location UTM33: (-10977, 6559806)

## How to use

Run in the following order:
1. `tracer_correct_8h.py`: Applies mass corrections to tracer concentrations based on real-world fluxes and discharges, then saves corrected output.
2. `transects_8h.py`: Takes corrected output and plots time-depth transects of tracer concentrations at shell farm.
3. `cu_anim_8h.py`: Takes corrected output and generates animations of maximum Cu concentrations in the water column from different sources over time.

## Requirements

- Python 3.x
- `xarray`
- `numpy`
- `matplotlib`
- `contextily`
- `dask`
- `pandas`

Install dependencies with:

```bash
pip install xarray numpy matplotlib contextily dask pandas
```

## Notes

- Custom colormaps are used to highlight concentration ranges for Cu critical limits
- Locations and sources are hard-coded for Lysefjord

---

## Logic of `correct_tracer_concentration`

Rescales the model tracer concentrations so that the total mass matches the real-world release for each tracer.

### Function Arguments

```python
def correct_tracer_concentration(
    ds,                      # xarray.Dataset containing tracer fields
    tracer_name,             # str, name of tracer variable in ds
    vol,                     # np.ndarray, grid cell volumes [shape: (time, layer, node)]
    time,                    # np.ndarray, FVCOM time values [shape: (time,)]
    dummy_river_conc=None,   # float, dummy river concentration in µg/m³ (model value)
    dummy_river_discharge=None, # float, dummy river discharge in m³/s (model value)
    real_river_conc=None,    # float, real river concentration in µg/m³ (real-world value)
    real_river_discharge=None,  # float, real river discharge in m³/s (real-world value)
    dummy_mass_flux=None,    # float, dummy mass flux in µg/s (model value, for injection tracers)
    real_mass_flux=None,     # float, real mass flux in µg/s (real-world value, for injection tracers)
    release_start_time=None, # float, FVCOM time value when release starts
    release_duration_seconds=8*3600 # float, duration of pulse release in seconds (default 8 hours)
)
```

- **ds**: xarray.Dataset with tracer fields, dimensions: time, siglay (layer), node.
- **tracer_name**: string, name of tracer variable in ds.
- **vol**: numpy array, grid cell volumes, shape (time, layer, node), units: m³.
- **time**: numpy array, FVCOM time values, shape (time,).
- **dummy_river_conc**: float, dummy river concentration in µg/m³ (model value).
- **dummy_river_discharge**: float, dummy river discharge in m³/s (model value).
- **real_river_conc**: float, real river concentration in µg/m³ (real-world value).
- **real_river_discharge**: float, real river discharge in m³/s (real-world value).
- **dummy_mass_flux**: float, dummy mass flux in µg/s (model value, for injection tracers).
- **real_mass_flux**: float, real mass flux in µg/s (real-world value, for injection tracers).
- **release_start_time**: float, FVCOM time value when release starts.
- **release_duration_seconds**: float, duration of pulse release in seconds.

### Step-by-Step Logic

1. **Mode Selection**
   - If `dummy_mass_flux` and `real_mass_flux` are provided, use *flux mode* (for injection tracers).
   - If `dummy_river_conc`, `dummy_river_discharge`, and `real_river_conc` are provided, use *concentration/discharge mode* (for river tracers).
   - Raises error if neither mode is satisfied.

2. **Elapsed Time Calculation**
   - Compute seconds since release started for each timestep:
     ```python
     seconds_since_release = np.maximum((time - release_start_time) * 86400, 0)
     # shape: (time,)
     ```

3. **Target Mass Calculation**
   - **Flux mode:** For injection tracers, calculate the real-world mass released up to each timestep.
     ```python
     effective_time = np.minimum(seconds_since_release, release_duration_seconds)
     target_mass_discharged = real_mass_flux * effective_time
     # units: µg, shape: (time,)
     ```
   - **Concentration/discharge mode:** For river tracers, calculate the real-world mass released up to each timestep.
     ```python
     target_mass_discharged = real_river_conc * real_river_discharge * seconds_since_release
     # units: µg, shape: (time,)
     ```

4. **Model Mass Calculation**
   - For each timestep, sum the total mass of tracer in the model domain:
     ```python
     int_mass_dummy = (vol * ds[tracer_name]).sum(dim=('node', 'siglay')).values
     # units: µg, shape: (time,)
     # ds[tracer_name]: shape (time, layer, node), units: µg/L
     # vol: shape (time, layer, node), units: m³
     # Multiplying gives µg/L * m³ = µg * 1000 (since 1 m³ = 1000 L)
     ```

5. **Correction Factor Calculation**
   - For each timestep, compute the ratio of real-world mass to model mass:
     ```python
     correction_factor = np.where(int_mass_dummy > 0, target_mass_discharged / int_mass_dummy, 1.0)
     # shape: (time,)
     # If model mass is zero, use factor 1.0 (no correction)
     correction_factor[0] = 1.0
     ```

6. **Apply Correction**
   - For each timestep, adjust the tracer concentration in every grid cell:
     ```python
     for t in range(n_time):
         field = ds[tracer_name][t].values           # shape: (layer, node), units: µg/L
         volume_t = vol[t]                           # shape: (layer, node), units: m³
         corrected_mass = field * volume_t * correction_factor[t]
         tracer_conc[t] = corrected_mass / volume_t / 1000  # shape: (layer, node), units: µg/L
     ```
   - The division by 1000 converts from µg/m³ to µg/L.

7. **Return**
   - Returns `tracer_conc`, numpy array of corrected concentrations, shape (time, layer, node), units: µg/L.

### Summary Formula

For each timestep and grid cell:
```
corrected_concentration = (original_concentration * cell_volume * correction_factor) / cell_volume / 1000
```
where
```
correction_factor = target_mass / model_mass
```

### Output

- The output array has the same shape as the input tracer field: (time, layer, node).
- Units are always µg/L.

This ensures the sum of corrected concentrations matches the real-world mass released for each tracer at each timestep.
This ensures the sum of corrected concentrations matches the real-world mass released for each tracer at each timestep.
