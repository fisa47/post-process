import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import contextily as ctx
import os
import pandas as pd
from datetime import datetime
from matplotlib.ticker import FixedLocator, FormatStrFormatter

origin = datetime(1858, 11, 17)

# Load datasets
shell_farm = (-9937.5, 6567670.1)
ds_all = xr.open_dataset('../output/lysefjord_tracers2_corrected_2h.nc', decode_times=False)
ds_output = xr.open_dataset('../output/lysefjord_0014.nc', decode_times=False).isel(time=-1)

ds_all['sum_cu'] = ds_all['river_tracer_c_corrected'] + ds_all['tracer2_c_corrected'] + ds_all['tracer4_c_corrected'] + ds_all['tracer6_c_corrected'] 

# Grid and time
area = ds_output['art1'].values
depth = ds_output['h'].values
surface_elevation = ds_all['zeta'].values
thickness = ds_output['siglev'].values
time = ds_all.time.values
tri = ds_output['nv'].values.T - 1
x, y = ds_output['x'], ds_output['y']

conc_sum = ds_all['sum_cu'].max(dim='siglay').values  # shape: (time, node)
conc_adnoy = ds_all['tracer2_c_corrected'].max(dim='siglay').values
conc_oltenvik = ds_all['tracer4_c_corrected'].max(dim='siglay').values
conc_gratness = ds_all['tracer6_c_corrected'].max(dim='siglay').values
conc_river = ds_all['river_tracer_c_corrected'].max(dim='siglay').values

# 1) Build a 512-entry colormap: 256 gray shades → 256 orange
n = 256
# gray ramp from white to black
gray_ramp   = plt.cm.gray(np.linspace(0, 1, n))

# flat orange for the next 256
orange_flat = np.tile(mcolors.to_rgba("#FFA500"), (n, 1))

# stack them into one ListedColormap
colors = np.vstack((gray_ramp, orange_flat))
cmap   = mcolors.ListedColormap(colors)

# 2) Anything above 5.2 - dark red
cmap.set_over("#8B0000")

# 3) Normalize 0 - 5.2 (so 2.6 is halfway in the colormap)
norm = mcolors.Normalize(vmin=0, vmax=5.2, clip=False)


### BIG ###

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import contextily as ctx

def animate_concentration(
    conc,            # ndarray, shape (nt, nnode)
    time,            # 1D array (days since start), length nt
    x, y,            # node coordinates
    shell_farm,      # (x,y) tuple
    sources,         # dict with source names and their (x,y) tuples
    output_path,     # "plots/Cu_big.mp4"
    basemap_crs="EPSG:32633",
    basemap_source=ctx.providers.OpenStreetMap.Mapnik,
    xlim=(-20000, 0),
    ylim=(6.55e6, 6.575e6),
    fps=5,
    dpi=200
):
    """Animate a scatter of conc over (x,y) with your special colormap."""
    # 1) Build colormap
    n = 256
    gray_ramp   = plt.cm.gray_r(np.linspace(0, 1, n))
    orange_flat = np.tile(mcolors.to_rgba("#FFA500"), (n, 1))
    colors = np.vstack((gray_ramp, orange_flat))
    cmap = mcolors.ListedColormap(colors)
    cmap.set_over("#8B0000")

    norm = mcolors.Normalize(vmin=0, vmax=5.2, clip=False)

    # 2) convert numeric days -> real timestamps
    #    time is days since origin
    real_times = pd.to_datetime(time, unit="D", origin=origin)

    # 2) Figure & initial scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        x, y,
        c=conc[0],
        cmap=cmap,
        norm=norm,
        s=3,
    )

    # Shell farm
    ax.scatter(*shell_farm, color='tomato', s=10)
    ax.annotate("Shell farm", shell_farm)

    # sources
    for name, (sx, sy) in sources.items():
        ax.scatter(sx, sy, color='blue', s=10)
        ax.annotate(name, (sx, sy), fontsize=8, ha='right')

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ctx.add_basemap(ax, crs=basemap_crs, source=basemap_source)

    title = ax.set_title("")

    # 3) Colorbar
    cbar = fig.colorbar(
        sc, ax=ax,
        extend='max',
        fraction=0.02, pad=0.04
    )
    ticks = [0, 1, 2, 2.6, 5.2]
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    cbar.ax.set_yticklabels(['0', '1', '2', '2.6', '>5.2'])
    cbar.set_label('Concentration (µg/L)')

    # 4) Update function
    def update(frame):
        sc.set_array(conc[frame])
        ts = real_times[frame]
        title.set_text(ts.strftime("Time: %Y-%m-%d %H:%M"))
        return sc, title

    # 5) Animate and save
    ani = animation.FuncAnimation(
        fig, update,
        frames=range(0, conc.shape[0], 1),
        blit=False, repeat=False
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
    plt.close(fig)
    print(f"Animation saved to {output_path}")

animate_concentration(
    conc=conc_sum,
    time=time,
    x=x, y=y,
    shell_farm=shell_farm,
    sources={
        'Lastabotn': (-10115, 6566772),
        'Ådnøy SØ': (-15715, 6565218),
        'Oltenvik': (-11046, 6557939),
        'Gråtness': (-10977, 6559806)
    },
    output_path="plots/Cu_sum.mp4"
)

animate_concentration(
    conc=conc_adnoy,
    time=time,
    x=x, y=y,
    shell_farm=shell_farm,
    sources={
        'Ådnøy SØ': (-15715, 6565218),  
    },
    output_path="plots/Cu_adnoy.mp4"
)

animate_concentration(
    conc=conc_oltenvik,
    time=time,
    x=x, y=y,
    shell_farm=shell_farm,
    sources={
        'Oltenvik': (-11046, 6557939),
    },
    output_path="plots/Cu_oltenvik.mp4"
)

animate_concentration(
    conc=conc_gratness,
    time=time,
    x=x, y=y,
    shell_farm=shell_farm,
    sources={
        'Gråtness': (-10977, 6559806),
    },
    output_path="plots/Cu_gratness.mp4"
)

animate_concentration(
    conc=conc_river,
    time=time,
    x=x, y=y,
    shell_farm=shell_farm,
    sources={
        'Lastabotn': (-10115, 6566772),
    },
    output_path="plots/Cu_river.mp4"
)
