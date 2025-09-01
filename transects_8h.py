import xarray as xr
import matplotlib.pyplot as plt
import contextily as ctx
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, FormatStrFormatter


fvcom_origin = datetime(1858, 11, 17)
shell_farm = (-9937.5, 6567670.1)

# Load dataset
ds_all = xr.open_dataset('/Users/Admin/Documents/scripts/fvcom-work/Lysefjord/output/lysefjord_tracers_corrected_2h_M2.nc', decode_times=False)
time = ds_all.time.values

time_start = 58344 # 2018-08-14 0:00:00

# Define tracer-specific parameters
conv = 1e6/3.6  # from kg/hr to ug/s
# Lastabotn:
# Mass flux = 0.00064 g/s
# Discharge = 40 L/s
tracers = [
   {'name': 'river_tracer_c', 'title': 'Lastabotn Cu', 'pipe_discharge': 10.0/60,
    'pipe_concentration': 10, 'real_c': 16*1e3, 'real_discharge': 0.01,
    'release_start': time[0], 'loc': (-10115, 6566772), 'vmin': 0, 'vmax': 0.001},

   {'name': 'tracer2_c', 'title': 'Ådnøy SØ 8 hours, Cu',
    'mass_flux_ug_per_s': 70, 'real_mass_flux_ug_per_s': 2*650/8*conv,
    'release_start':  58344.166667,  # 2018-08-14 4:00:00
    'loc': (-15715, 6565218), 'pipe_ind': 37284, 'vmin': 0, 'vmax': 1},

   {'name': 'tracer4_c', 'title': 'Oltenvik 8 hours, Cu',
    'mass_flux_ug_per_s': 90, 'real_mass_flux_ug_per_s': 2*262/8*conv,
    'release_start': 58344.375000,  # 2018-08-14 9:00:00
    'loc': (-11046, 6557939), 'pipe_ind': 49112, 'vmin': 0, 'vmax': 1},

   {'name': 'tracer4_c_copy', 'title': 'Oltenvik 8 hours, Tralapyril', 
    'mass_flux_ug_per_s': 90, 'real_mass_flux_ug_per_s': 10.4/8*conv,
    'release_start': 58344.375000,  # 2018-08-14 9:00:00
    'loc': (-11046, 6557939), 'pipe_ind': 49112, 'vmin': 0, 'vmax': 1},

   {'name': 'tracer6_c', 'title': 'Gråtnes 8 hours',
    'mass_flux_ug_per_s': 80, 'real_mass_flux_ug_per_s': 2*356/8*conv,
    'release_start': 58344.375000,  # 2018-08-14 9:00:00
    'loc': (-10977, 6559806), 'pipe_ind': 48198, 'vmin': 0, 'vmax': 11},
]

# Grid data
# Get it from actual output file
ds_output = xr.open_dataset('/Users/Admin/Documents/scripts/fvcom-work/Lysefjord/output/jul14/lysefjord_0001.nc', decode_times=False).isel(time=-1)
area = ds_output['art1'].values
depth = ds_output['h'].values
surface_elevation = ds_all['zeta'].values
thickness = ds_output['siglev'].values

tri = ds_output['nv'].values.T - 1
x, y = ds_output['x'], ds_output['y']

# Build the 0 – 2.6 gray -> 2.6 – 5.2 orange -> >5.2 dark-red colormap
n = 256
gray_ramp   = plt.cm.gray_r(np.linspace(0, 1, n))             # 256‐step gray
orange_flat = np.tile(mcolors.to_rgba("#FFA500"), (n, 1))   # 256‐step flat orange

colors = np.vstack((gray_ramp, orange_flat))
cmap   = mcolors.ListedColormap(colors)
cmap.set_over("#8B0000")                                    # dark red for >5.2

norm = mcolors.Normalize(vmin=0, vmax=5.2, clip=False)
# ──────────────────────────────────────────────────────────────

# === Transect plot at Shell Farm (node = 51016) ===
print("Plotting Cu time-depth transect at Shell Farm (node 51016)...")

shell_node = 51016
sum_depth_conc = None

# Sum concentrations from all tracers at the shell farm node
for tracer in tracers:
    if tracer['name'] in ['river_tracer_c', 'tracer2_c', 'tracer4_c', 'tracer6_c']:
        tracer_name = tracer['name']
        conc_at_node = ds_all[f'{tracer_name}_corrected'].sel(node=shell_node).values  # shape: (time, siglay, node)
        if sum_depth_conc is None:
            sum_depth_conc = conc_at_node
        else:
            sum_depth_conc += conc_at_node

print('tracer: ', sum_depth_conc.shape)

# Get corresponding depths for each sigma layer
depth_at_node = ds_all['siglay'].sel(node=shell_node).values * ds_all['h'].sel(node=shell_node).isel(time=0).values  # shape: (siglay,)
print('depth_at_node: ', depth_at_node.shape)
depth_matrix = np.tile(depth_at_node, (len(time), 1))  # shape: (time, siglay)
print('depth_matrix: ', depth_matrix.shape)

# Convert FVCOM time to datetime
time_days = (time - time_start)  # days since start

def plot_transect(tracer_depth_conc,
                  depth_at_node,
                  time_days,
                  cmap,
                  norm,
                  title,
                  fname):
    """
    Plot the transect of tracer concentration over time and depth.
    """

    fig, ax = plt.subplots(figsize=(8, 4))

    # levels between 0 – 5.2
    levels = np.linspace(0, 5.2, 101)

    cf = ax.contourf(
        time_days,
        depth_at_node,
        tracer_depth_conc.T,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend='max'
    )

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, extend='max', fraction=0.02, pad=0.04)
    ticks = [0, 1, 2, 2.6, 5.2]
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    cbar.ax.set_yticklabels(['0', '1', '2', '2.6', '>5.2'])
    cbar.set_label("μg/L")

    ax.set_xlim(0, 21)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_title(title)
    ax.set_xlabel("Days Since Start")
    ax.set_ylabel("Depth (m)")
    ax.grid(True)

    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)

    fig.savefig(f"plots/{fname}", dpi=200)
    print(title, "saved to plots/" + fname)
    plt.close()

plot_transect(sum_depth_conc,
                  depth_at_node, time_days, cmap, norm,
                  "Total Concentration of Cu (μg/L) at Shell Farm from all sources",
                  "shell_farm_Cu_sum_transect.png")

plot_transect(ds_all[f'river_tracer_c_corrected'].sel(node=shell_node).values,
                  depth_at_node, time_days, cmap, norm,
                  "Concentration of Cu (μg/L) at Shell Farm from Lastabotn",
                  "shell_farm_Cu_Lastabotn_transect.png")

plot_transect(ds_all[f'tracer2_c_corrected'].sel(node=shell_node).values,
                  depth_at_node, time_days, cmap, norm,
                  "Concentration of Cu (μg/L) at Shell Farm from Ådnøy SØ 8 hours release",
                  "shell_farm_Cu_Adnoy8h_transect.png")

plot_transect(ds_all[f'tracer4_c_corrected'].sel(node=shell_node).values,
                  depth_at_node, time_days, cmap, norm,
                  "Concentration of Cu (μg/L) at Shell Farm from Oltenvik 8 hours release",
                  "shell_farm_Cu_Oltenvik8h_transect.png")

plot_transect(ds_all[f'tracer6_c_corrected'].sel(node=shell_node).values,
                  depth_at_node, time_days, cmap, norm,
                  "Concentration of Cu (μg/L) at Shell Farm from Gråtness 8 hours release ",
                  "shell_farm_Cu_Gratness8h_transect.png")