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

    # Calculate wind speed in km/h
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort Surface -> Space
    inds = p.argsort()[::-1]
    p_hpa, t, td, wind_speed_val = p[inds].to(units.hPa), t[inds], td[inds], wind_speed[inds]

    # 3. Figure Setup (Tall aspect ratio)
    fig = plt.figure(figsize=(12, 14))
    
    # Define boxes: Main Sounding (Skew-T) and Wind Speed (Linear)
    # rect = [left, bottom, width, height]
    ax1_box = [0.1, 0.1, 0.65, 0.8]
    ax2_box = [0.75, 0.1, 0.15, 0.8]

    # --- PANEL 1: MAIN SOUNDING (SKEWED) ---
    # rotation=45 provides the sharp "lean" seen in MeteoSwiss charts
    skew = SkewT(fig, rotation=45, rect=ax1_box)
    
    # Range adjustments: x-axis expanded for the steeper lean
    skew.ax.set_xlim(-40, 50) 
    skew.ax.set_ylim(1020, 400) # Surface to ~7km
    skew.ax.set_aspect('auto')

    # Plot Sounding Data
    skew.plot(p_hpa, t, 'red', linewidth=2.5, label='Temperature')
    skew.plot(p_hpa, td, 'red', linewidth=1.5, linestyle='--', label='Dewpoint')
    
    # Helper Lines (Standard Physics)
    skew.plot_dry_adiabats(alpha=0.2, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.2, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.2, color='green', linestyle=':', linewidth=0.7)

    # --- PANEL 2: WIND SPEED (km/h) ---
    ax2 = fig.add_axes(ax2_box)
    ax2.set_yscale('log') # Log scale to match SkewT altitude logic
    ax2.set_ylim(1020, 400)
    ax2.set_xlim(0, 100) # 0 to 100 km/h
    
    ax2.plot(wind_speed_val, p_hpa, 'red', linewidth=2)
    ax2.fill_betweenx(p_hpa, 0, wind_speed_val, color='red', alpha=0.05)

    # --- SHARED ALTITUDE LABELS & GRID ---
    km_all = np.arange(0, 8.5, 0.5) 
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    
    # Main Plot Y-axis (Altitude only, NO Hectopascals)
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_minor_locator(NullLocator()) 
    skew.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])}" if km_all[pos] % 1 == 0 else ""))
    
    # Wind Plot Y-axis (Hidden labels, shared ticks)
    ax2.yaxis.set_major_locator(FixedLocator(p_levels))
    ax2.set_yticklabels([]) 
    
    # Draw horizontal helper lines across the entire chart area
    for level in p_levels:
        skew.ax.axhline(level, color='black', alpha=0.1, linewidth=0.5)
        ax2.axhline(level, color='black', alpha=0.1, linewidth=0.5)

    # --- CLEANUP & PILOT STYLING ---
    skew.ax.set_ylabel("km", loc='top', rotation=0, labelpad=-20)
    skew.ax.set_xlabel("°C")
    ax2.set_xlabel("km/h")
    
    # Wind ticks
    ax2.set_xticks([0, 20, 40, 60, 80, 100])
    ax2.grid(True, axis='x', alpha=0.2)

    # Title
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d-%m-%Y %H:%M')
    fig.suptitle(f"Sounding Payerne | {ref_time_str} UTC", fontsize=16, y=0.95)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: MeteoSwiss-style tall emagram with km/h wind panel generated.")

if __name__ == "__main__":
    main()
