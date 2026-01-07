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
    fig = plt.figure(figsize=(12, 12))
    
    # We use a slight rotation (30) to keep lapse rates clear without being too squashed
    skew = SkewT(fig, rotation=30)
    
    # --- PARAGLIDING SCALE ADJUSTMENTS ---
    # Set Limits: Expanded left (colder) and tight bottom (1020hPa) to remove gap
    # ICON-CH1 surface is usually around 1000-1015hPa for lowlands
    skew.ax.set_ylim(1020, 400) 
    skew.ax.set_xlim(-40, 35)   

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Barbs on the far right
    skew.plot_barbs(p_hpa[::2], u[::2], v[::2], xloc=1.02)
    
    # 5. Background Adiabats (Thermal Strength Reference)
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linewidth=1)
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linewidth=1)
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':')

    # 6. Altitude Tick Optimization
    # We define standard altitude levels in meters
    altitudes_m = np.arange(0, 8000, 1000)
    # Convert those altitudes back to pressure levels (std atmosphere) for placement
    pressure_ticks = mpcalc.height_to_pressure_std(altitudes_m * units.m).to(units.hPa).m
    
    skew.ax.yaxis.set_major_locator(FixedLocator(pressure_ticks))
    skew.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(mpcalc.pressure_to_height_std(x * units.hPa).to(units.km).m)} km"))
    
    # Clean up labels
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # 7. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=16, pad=20)
    skew.ax.legend(loc='upper left', frameon=True)
    
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: Cleaned linear-height Skew-T generated.")

if __name__ == "__main__":
    main()
