import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
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

    # Calculate Altitude (Z) using standard atmosphere pressure-to-height
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort data for clean plotting
    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds].m, t[inds].m, td[inds].m, wind_speed[inds].m
    p_plot = p[inds]

    # 3. Figure Setup
    fig = plt.figure(figsize=(14, 12))
    
    # --- ADJUSTABLE SKEW LOGIC ---
    ax1 = fig.add_axes([0.1, 0.1, 0.55, 0.8])
    
    # Increasing this angle (e.g., 50-60) will tilt isotherms more horizontally
    # Reducing it (e.g., 30) makes them more vertical
    skew_angle = 55 
    base_trans = ax1.transData
    skew = transforms.Affine2D().skew_deg(skew_angle, 0)
    t_skew = skew + base_trans

    # --- PANEL 1: THERMO DIAGRAM ---
    ax1.set_ylim(0, 8.5)   
    ax1.set_xlim(-30, 50) 
    
    # Helper Lines: Standard height/pressure reference
    z_ref = np.linspace(0, 9, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)

    # Draw Isotherms (Tilted)
    for temp in range(-60, 81, 10):
        ax1.plot([temp, temp], [0, 9], color='blue', alpha=0.08, linestyle='-', transform=t_skew, zorder=1)

    # Dry Adiabats (Potential Temperature lines)
    for theta in range(-20, 140, 10):
        theta_val = (theta + 273.15) * units.K
        t_adiabat = mpcalc.dry_lapse(p_ref, theta_val, 1000 * units.hPa).to(units.degC).m
        ax1.plot(t_adiabat, z_ref.m, color='brown', alpha=0.15, linewidth=1, transform=t_skew, zorder=2)

    # Mixing Ratio Lines (FIXED Calculation)
    for w in [1, 2, 4, 7, 10, 16, 24, 32]:
        w_val = w * units('g/kg')
        # Correct sequence: Mixing Ratio -> Vapor Pressure -> Dewpoint
        e_w = mpcalc.vapor_pressure(p_ref, w_val)
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        ax1.plot(t_w, z_ref.m, color='green', alpha=0.15, linestyle=':', transform=t_skew, zorder=2)

    # Plot Actual Sounding
    ax1.plot(t_plot, z_plot, 'red', linewidth=3, label='Temp', transform=t_skew, zorder=5)
    ax1.plot(td_plot, z_plot, 'green', linewidth=3, label='Dewpoint', transform=t_skew, zorder=5)

    ax1.set_ylabel("Standard Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.2)
    ax1.legend(loc='upper right')

    # --- PANEL 2: WIND SPEED ---
    ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='blue', alpha=0.1)
    
    ax2.set_ylim(0, 8.5)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Wind Speed (km/h)", fontsize=12)
    ax2.set_yticklabels([]) 
    ax2.grid(True, alpha=0.2)

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Custom Atmospheric Profile | Payerne | {ref_time_str} UTC", fontsize=16)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Success: Generated custom profile with scientific reference lines.")

if __name__ == "__main__":
    main()
