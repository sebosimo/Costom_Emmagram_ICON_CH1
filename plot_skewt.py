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

    # 3. Visualization Setup (Restored to tall aspect ratio 10x12)
    fig = plt.figure(figsize=(10, 12))
    skew = SkewT(fig, rotation=30)
    
    # --- SCALE ADJUSTMENTS ---
    # Shifted further left to -60C to avoid dewpoint cutoff
    skew.ax.set_xlim(-60, 35)
    # Tight bottom at 1020 hPa to ground the plot, cut at 400 hPa
    skew.ax.set_ylim(1020, 400) 

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Position barbs cleanly on the right
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3], xloc=1.02)
    
    # 5. Supporting Adiabats
    skew.plot_dry_adiabats(alpha=0.1, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.1, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.1, color='green', linestyle=':')

    # 6. FIXED Altitude Ticks (Every 0.5 km)
    # Generate list of heights: 0, 0.5, 1.0, 1.5... up to 8km
    km_all = np.arange(0, 8.5, 0.5) 
    # Convert those heights to the standard pressure levels where they belong
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    
    # Force the Y-axis to use these specific pressure points
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_minor_locator(NullLocator()) 
    
    # Custom Formatter: Label 1km, 2km etc. Leave 0.5km lines blank but visible
    def km_formatter(x, pos):
        if pos < len(km_all):
            val = km_all[pos]
            if val % 1 == 0: # Only whole numbers
                return f"{int(val)} km"
        return "" 

    skew.ax.yaxis.set_major_formatter(FuncFormatter(km_formatter))
    
    # Grid lines at every 0.5km tick for easy horizontal reading
    skew.ax.grid(True, which='major', axis='y', color='black', alpha=0.1, linestyle='-')
    
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # 7. Metadata and Title
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=15, pad=20)
    skew.ax.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: Tall-format plot with 0.5km helper lines generated.")

if __name__ == "__main__":
    main()
