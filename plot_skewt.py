import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob
import numpy as np

CACHE_DIR = "cache_data"

def main():
    # 1. Load Data
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)
    
    # 2. Extract values and assign units
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    u, v = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # Sort Surface -> Space
    inds = p.argsort()[::-1]
    p_hpa, t, td, u, v = p[inds].to(units.hPa), t[inds], td[inds], u[inds], v[inds]

    # 3. Visualization Setup
    # We use a very specific tall figure size (8x14)
    fig = plt.figure(figsize=(8, 14))
    
    # We use 'rect' to give the plot its own defined space inside the figure
    # This creates the vertical "pillar" look and prevents the right side from clipping
    skew = SkewT(fig, rotation=30, rect=(0.12, 0.1, 0.70, 0.85))
    
    # Force auto aspect to ensure it fills our tall 'rect'
    skew.ax.set_aspect('auto')
    
    # --- RANGE ADJUSTMENTS ---
    # Left cut: -40 (enough for dry dewpoints) 
    # Right expand: 45 (enough for hot summer ground temps)
    skew.ax.set_xlim(-40, 45)
    
    # Vertical range: 1020hPa to 400hPa (~7km)
    skew.ax.set_ylim(1020, 400) 

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Place barbs strictly at the edge of the plot area
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3], xloc=1.0)
    
    # 5. Supporting Adiabats
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':')

    # 6. ALTITUDE LABELS & GRID (Every 0.5 km)
    km_all = np.arange(0, 8.5, 0.5) 
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_minor_locator(NullLocator()) 
    
    def km_formatter(x, pos):
        # Identify exact km values from our list to prevent mismatch
        if pos < len(km_all):
            val = km_all[pos]
            if val % 1 == 0:
                return f"{int(val)} km"
        return "" 

    skew.ax.yaxis.set_major_formatter(FuncFormatter(km_formatter))
    
    # Stronger helper lines for every 0.5km step
    skew.ax.grid(True, which='major', axis='y', color='black', alpha=0.15, linestyle='-')
    
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # 7. Metadata and Title
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=15, pad=30)
    skew.ax.legend(loc='upper left', frameon=True)

    # 8. Save with enough padding for wind barbs
    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Tall-format summer-ready plot generated.")

if __name__ == "__main__":
    main()
