import matplotlib.pyplot as plt
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

    # Calculate Standard Altitude (km)
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort for plotting
    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds].m, t[inds].m, td[inds].m, wind_speed[inds].m
    p_plot = p[inds]

    # --- SKEW CONFIGURATION ---
    # This factor controls the tilt. 
    # At 0, isotherms are vertical. At 5-10, they tilt like a standard Skew-T.
    SKEW_FACTOR = 8 

    def skew_x(temp, height):
        """Returns the skewed X-coordinate for a given temperature and height."""
        return temp + (height * SKEW_FACTOR)

    # 3. Figure Setup
    fig = plt.figure(figsize=(14, 12))
    ax1 = fig.add_axes([0.1, 0.1, 0.55, 0.8])
    
    # Define Plot Limits (adjusted for skew)
    z_max = 8.5
    ax1.set_ylim(0, z_max)
    ax1.set_xlim(skew_x(-30, 0), skew_x(40, z_max)) # Keep view centered on relevant temps

    # --- DRAW HELPER LINES ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)

    # 1. Tilted Isotherms
    for temp in range(-60, 81, 10):
        # We calculate two points for the line: bottom and top
        x_start = skew_x(temp, 0)
        x_end = skew_x(temp, z_max)
        ax1.plot([x_start, x_end], [0, z_max], color='blue', alpha=0.1, linestyle='-', zorder=1)

    # 2. Dry Adiabats
    for theta in range(-20, 140, 10):
        theta_val = (theta + 273.15) * units.K
        t_adiabat = mpcalc.dry_lapse(p_ref, theta_val, 1000 * units.hPa).to(units.degC).m
        ax1.plot(skew_x(t_adiabat, z_ref.m), z_ref.m, color='brown', alpha=0.15, linewidth=1, zorder=2)

    # 3. Mixing Ratio Lines
    for w in [1, 2, 4, 7, 10, 16, 24, 32]:
        w_val = w * units('g/kg')
        e_w = mpcalc.vapor_pressure(p_ref, w_val)
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        ax1.plot(skew_x(t_w, z_ref.m), z_ref.m, color='green', alpha=0.15, linestyle=':', zorder=2)

    # --- PLOT DATA ---
    ax1.plot(skew_x(t_plot, z_plot), z_plot, 'red', linewidth=3, label='Temp', zorder=5)
    ax1.plot(skew_x(td_plot, z_plot), z_plot, 'green', linewidth=3, label='Dewpoint', zorder=5)

    # --- FORMATTING ---
    # Custom Ticks: We want the labels to show the REAL temperature, 
    # but they must be placed at the SKEWED X-position on the bottom axis (z=0).
    temp_ticks = np.arange(-40, 51, 10)
    ax1.set_xticks([skew_x(t, 0) for t in temp_ticks])
    ax1.set_xticklabels(temp_ticks)

    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.2)
    ax1.legend(loc='upper right')

    # --- PANEL 2: WIND SPEED ---
    ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='blue', alpha=0.1)
    ax2.set_ylim(0, z_max)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Wind Speed (km/h)", fontsize=12)
    ax2.set_yticklabels([]) 
    ax2.grid(True, alpha=0.2)

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Skewed Atmospheric Profile | Payerne | {ref_time_str} UTC", fontsize=16)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print(f"Success: Generated manual skewed plot with SKEW_FACTOR={SKEW_FACTOR}")

if __name__ == "__main__":
    main()
