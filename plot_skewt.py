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

    inds = p.argsort()[::-1]
    p_hpa, t, td, u, v = p[inds].to(units.hPa), t[inds], td[inds], u[inds], v[inds]

    # 3. Visualization Setup
    # Create a tall figure
    fig = plt.figure(figsize=(9, 12)) 
    
    # IMPORTANT: We define a specific box for the plot to live in.
    # This ensures it stays tall even if we cut the top range.
    skew = SkewT(fig, rotation=30, rect=(0.1, 0.1, 0.75, 0.85))
    
    # --- SCALE ADJUSTMENTS ---
    skew.ax.set_xlim(-60, 35)
    # Surface to ~7km (400 hPa)
    skew.ax.set_ylim(1020, 400) 

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # 5. FIXED Altitude Ticks (0.5 km logic)
    km_all = np.arange(0, 8.5, 0.5) 
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_minor_locator(NullLocator()) 
    
    # Format labels: only show full km values to avoid clutter
    def km_formatter(x, pos):
        # pos is the index in our FixedLocator (km_all)
        if pos < len(km_all):
            val = km_all[pos]
            if val % 1 == 0:
                return f"{int(val)} km"
        return "" 

    skew.ax.yaxis.set_major_formatter(FuncFormatter(km_formatter))
    
    # 6. Supporting Reference Lines
    skew.plot_dry_adiabats(alpha=0.1, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.1, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.1, color='green', linestyle=':')
    
    # Force horizontal lines at every 0.5km step
    skew.ax.grid(True, which='major', axis='y', color='black', alpha=0.1)
    
    # 7. Professional Styling
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # Position barbs cleanly on the right of the tall box
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3], xloc=1.02)
    
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=14, pad=20)
    skew.ax.legend(loc='upper left', frameon=True)

    # 8. Save with explicit dpi and no aggressive cropping
    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Tall aspect ratio restored with focused range.")

if __name__ == "__main__":
    main()
