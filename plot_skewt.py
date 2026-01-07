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
    fig = plt.figure(figsize=(10, 14))
    
    # Define exact boxes for the two plots (Left, Bottom, Width, Height)
    # No gap: ax1 ends at 0.7, ax2 starts at 0.7
    ax1_box = [0.1, 0.1, 0.6, 0.8]
    ax2_box = [0.7, 0.1, 0.2, 0.8]

    # --- PANEL 1: MAIN SOUNDING ---
    skew = SkewT(fig, rotation=30, rect=ax1_box)
    skew.ax.set_xlim(-50, 45) # Expanded left for lines/dewpoint
    skew.ax.set_ylim(1020, 350) # Surface to ~8km
    skew.ax.set_aspect('auto')

    # Plot Sounding Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Helper Lines (Standard Physics)
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linewidth=1)
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linewidth=1)
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':', linewidth=0.8)

    # --- PANEL 2: WIND SPEED (km/h) ---
    ax2 = fig.add_axes(ax2_box)
    ax2.set_yscale('log') # Must match main plot to align height lines
    ax2.set_ylim(1020, 350)
    ax2.set_xlim(0, 80)
    
    ax2.plot(wind_speed_val, p_hpa, 'blue', linewidth=2.5)
    ax2.fill_betweenx(p_hpa, 0, wind_speed_val, color='blue', alpha=0.1)

    # --- SHARED ALTITUDE LABELS & HELPER LINES ---
    km_all = np.arange(0, 9, 0.5) 
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    
    # Apply to Main Plot
    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])} km" if km_all[pos] % 1 == 0 else ""))
    
    # Apply to Wind Plot (Hide labels, keep ticks for alignment)
    ax2.yaxis.set_major_locator(FixedLocator(p_levels))
    ax2.set_yticklabels([]) 
    
    # Extended Horizontal Helper Lines across BOTH panels
    for level in p_levels:
        # Draw on main plot
        skew.ax.axhline(level, color='black', alpha=0.1, linewidth=0.8)
        # Draw on wind plot
        ax2.axhline(level, color='black', alpha=0.1, linewidth=0.8)

    # --- CLEANUP & STYLING ---
    # Remove all hPa/Pressure mentions
    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    ax2.set_xlabel("Wind (km/h)")
    
    # Add wind ticks every 20 km/h
    ax2.set_xticks([0, 20, 40, 60, 80])
    ax2.grid(True, axis='x', alpha=0.2)

    # Metadata Title
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Payerne Sounding | {ref_time_str} UTC", fontsize=16, y=0.94)
    skew.ax.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Integrated Emagram and Wind Profile generated.")

if __name__ == "__main__":
    main()
