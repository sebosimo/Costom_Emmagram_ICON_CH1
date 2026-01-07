import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
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
    # 16:9 wide aspect ratio for better horizontal resolution
    fig = plt.figure(figsize=(16, 9))
    skew = SkewT(fig, rotation=30)
    
    # --- SCALE ADJUSTMENTS ---
    # Expanded left limit to -50C for dewpoint breathing room
    skew.ax.set_xlim(-50, 35)
    # 1020 hPa removes the bottom gap; 400 hPa focuses on flight levels
    skew.ax.set_ylim(1020, 400) 

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Barbs on the far right (xloc=1.05)
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3], xloc=1.05)
    
    # 5. Background Reference Lines
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':')

    # 6. FIXED Altitude Ticks (0, 1, 2... km)
    # Define heights in kilometers
    km_levels = np.arange(0, 8, 1) 
    # Convert height to standard pressure for exact placement
    p_levels = mpcalc.height_to_pressure_std(km_levels * units.km).to(units.hPa).m
    
    # Force EXACT positions and labels
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if pos < len(km_levels) else ""))
    
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # 7. Metadata and Title
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=18, pad=25)
    skew.ax.legend(loc='upper left', frameon=True, fontsize=12)
    
    # Subtle horizontal grid lines matching the km levels
    skew.ax.grid(True, which='major', axis='y', color='black', alpha=0.05)

    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: Updated plot with corrected km labels generated.")

if __name__ == "__main__":
    main()
