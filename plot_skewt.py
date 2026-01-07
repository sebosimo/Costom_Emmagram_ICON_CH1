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
    
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # Calculate Standard Altitude (km)
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort and filter for 0-7 km
    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds].m, t[inds].m, td[inds].m, wind_speed[inds].m
    
    z_max = 7.0
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot, wind_plot = z_plot[mask], t_plot[mask], td_plot[mask], wind_plot[mask]
    p_plot = p[inds][mask]

    # --- SKEW CONFIGURATION ---
    SKEW_FACTOR = 12 
    def skew_x(temp, height):
        return temp + (height * SKEW_FACTOR)

    # 3. Figure Setup
    # gridspec_kw={'wspace': 0} removes the gap between plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True, 
                                   gridspec_kw={'width_ratios': [4, 1], 'wspace': 0})
    
    ax1.set_ylim(0, z_max)
    
    # CUT SPACE: Set limits to only show -40 to +40 range at the surface
    # The left/right bounds are determined by the skew at the bottom (z=0)
    ax1.set_xlim(skew_x(-40, 0), skew_x(40, 0))

    # --- DRAW HELPER LINES ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)

    # 1. Horizontal Helper Lines (Constant Altitude) - Made prominent
    ax1.grid(True, axis='y', color='gray', alpha=0.3, linestyle='-', linewidth=0.8)

    # 2. Tilted Isotherms (Every 10°C)
    for temp in range(-60, 81, 10):
        ax1.plot([skew_x(temp, 0), skew_x(temp, z_max)], [0, z_max], 
                 color='blue', alpha=0.1, linestyle='-', zorder=1)

    # 3. Dry Adiabats
    for theta in range(-20, 140, 10):
        theta_val = (theta + 273.15) * units.K
        t_adiabat = mpcalc.dry_lapse(p_ref, theta_val, 1000 * units.hPa).to(units.degC).m
        ax1.plot(skew_x(t_adiabat, z_ref.m), z_ref.m, color='brown', alpha=0.15, linewidth=1, zorder=2)

    # 4. Mixing Ratio Lines
    for w in [1, 2, 4, 7, 10, 16, 24, 32]:
        e_w = mpcalc.vapor_pressure(p_ref, w * units('g/kg'))
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        ax1.plot(skew_x(t_w, z_ref.m), z_ref.m, color='green', alpha=0.15, linestyle=':', zorder=2)

    # --- PLOT DATA ---
    ax1.plot(skew_x(t_plot, z_plot), z_plot, 'red', linewidth=3, label='Temp', zorder=5)
    ax1.plot(skew_x(td_plot, z_plot), z_plot, 'green', linewidth=3, label='Dewpoint', zorder=5)

    # Formatting Temp Axis labels (Showing real -40 to 40 range)
    temp_ticks = np.arange(-40, 41, 10)
    ax1.set_xticks([skew_x(t, 0) for t in temp_ticks])
    ax1.set_xticklabels(temp_ticks)
    
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.legend(loc='upper right', frameon=True)

    # --- PANEL 2: WIND SPEED (Attached) ---
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='blue', alpha=0.1)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.grid(True, axis='both', alpha=0.2)
    
    # Clean up spines for the "seamless" look
    ax2.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Paragliding Sounding | Payerne | {ref_time_str} UTC", fontsize=16, y=0.95)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Success: Generated refined compact skewed plot with requested bounds.")

if __name__ == "__main__":
    main()
