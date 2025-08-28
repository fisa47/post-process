# Lysefjord FVCOM Post-Processing Scripts

## Input Data

The main input file is a merged NetCDF file created from FVCOM 2 hours restart files. This was necessary because FABM did not write the required tracer variables to the standard output files.
The example is for 8 hours release.

## Tracers

Below is a summary of the main tracers used in the simulations:

- **Lastabotn Cu (`river_tracer_c`)**  
    Copper released from the Lastabotn river.  
    - Discharge: 10 L/min (model river), 0.01 m³/s (real-world)  
    - Concentration: 10 µg/L (model river), 16 mg/m³ (real-world)  
    - Release start: beginning of the simulation 
    - Location: (-10115, 6566772)

- **Ådnøy SØ 8 hours, Cu (`tracer2_c`)**  
    Copper released at Ådnøy SØ over 8 hours.  
    - Dummy mass flux: 70 µg/s = 10 µg/s × 7 release levels
    - Real mass flux: 2 (doubled release) × 650 kg / 8 hr × conv µg/s 
    - Release start: Simulation time 58344.166667 (2018-08-14 04:00:00)  
    - Location: (-15715, 6565218)

- **Oltenvik 8 hours, Cu (`tracer4_c`)**  
    Copper released at Oltenvik over 8 hours.  
    - Dummy mass flux: 90 µg/s = 10 µg/s * 9 release levels
    - Real mass flux: 2 (doubled release) × 262 kg / 8 hr × conv µg/s  
    - Release start: Simulation time 58344.375000 (2018-08-14 09:00:00)  
    - Location: (-11046, 6557939)

- **Oltenvik 8 hours, Tralapyril (`tracer4_c_copy`)**  
    Tralapyril released at Oltenvik, same timing and location as copper.  
    - Dummy mass flux: 90 µg/s = 10 µg/s * 9 release levels 
    - Real mass flux: 10.4 kg / 8 hr × conv µg/s  
    - Release start: Simulation time 58344.375000  
    - Location: (-11046, 6557939)

- **Gråtnes 8 hours (`tracer6_c`)**  
    Substance released at Gråtnes over 8 hours.  
    - Dummy mass flux: 80 µg/s = 10 µg/s * 8 release levels 
    - Real mass flux: 2 (doubled release) × 356 kg / 8 hr × conv µg/s 
    - Release start: Simulation time 58344.375000 (2018-08-14 09:00:00)  
    - Location: (-10977, 6559806)

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
