
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
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

    # 3. Figure Setup (Wide canvas to fit two panels)
    fig = plt.figure(figsize=(12, 14))
    
    # Define two panels: [left, bottom, width, height]
    # Panel 1: Emagram (Main Sounding)
    ax1 = fig.add_axes([0.1, 0.08, 0.6, 0.85]) 
    # Panel 2: Wind Speed in km/h
    ax2 = fig.add_axes([0.75, 0.08, 0.15, 0.85])

    # --- PANEL 1: EMAGRAM ---
    # Log scale mimics linear altitude
    ax1.set_yscale('log')
    ax1.set_ylim(1020, 400)
    ax1.set_xlim(-30, 50)
    
    # Custom Background Lines (Standard Lapse Rate reference)
    # Drawing simple diagonal lines to represent the 'skew'
    for temp in range(-60, 80, 10):
        ax1.plot([temp, temp-40], [1020, 400], color='gray', alpha=0.1, linestyle='-', linewidth=0.5)

    # Plot Sounding
    ax1.plot(t, p_hpa, 'red', linewidth=3, label='Temperature')
    ax1.plot(td, p_hpa, 'green', linewidth=3, label='Dewpoint')
    
    # Altitude Labels for ax1
    km_all = np.arange(0, 8.5, 0.5)
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    ax1.yaxis.set_major_locator(FixedLocator(p_levels))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])} km" if km_all[pos] % 1 == 0 else ""))
    
    ax1.grid(True, which='major', axis='y', color='black', alpha=0.15)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.legend(loc='upper left')

    # --- PANEL 2: WIND SPEED ---
    ax2.set_yscale('log')
    ax2.set_ylim(1020, 400)
    ax2.set_xlim(0, 80) # Max 80 km/h wind
    
    # Plot Wind Speed Profile
    ax2.plot(wind_speed_val, p_hpa, 'blue', linewidth=2)
    ax2.fill_betweenx(p_hpa, 0, wind_speed_val, color='blue', alpha=0.1)
    
    # Formatting ax2
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.set_yticklabels([]) # Hide altitude labels on the second panel
    ax2.grid(True, which='major', axis='both', alpha=0.2)
    
    # Standard labels for wind speed
    ax2.set_xticks([0, 20, 40, 60, 80])

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Paragliding Emagram & Wind Profile | Payerne | {ref_time_str} UTC", fontsize=16)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Custom Emagram with km/h wind panel generated.")

if __name__ == "__main__":
    main()
