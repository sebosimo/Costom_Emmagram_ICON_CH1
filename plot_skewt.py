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

    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds].m, t[inds].m, td[inds].m, wind_speed[inds].m
    p_plot = p[inds]

    # 3. Figure Setup
    fig = plt.figure(figsize=(14, 12))
    
    # --- Transformation Logic ---
    ax1 = fig.add_axes([0.1, 0.1, 0.55, 0.8])
    
    # Skew transformation: 45 degrees
    # We use a 1:1 ratio for Temp(C) and Alt(km) to make the slant 45 deg
    skew_angle = 45 
    base_trans = ax1.transData
    skew = transforms.Affine2D().skew_deg(skew_angle, 0)
    t_skew = skew + base_trans

    # --- PANEL 1: CUSTOM SKEW-T ---
    ax1.set_ylim(0, 8)  
    ax1.set_xlim(-25, 45) 
    
    # Plot Isotherms (Corrected Fix)
    for temp in range(-50, 61, 10):
        ax1.plot([temp, temp], [0, 8], color='blue', alpha=0.1, linestyle='--', transform=t_skew, zorder=1)

    # Plot Sounding
    ax1.plot(t_plot, z_plot, 'red', linewidth=3, label='Temp', transform=t_skew, zorder=5)
    ax1.plot(td_plot, z_plot, 'green', linewidth=3, label='Dewpoint', transform=t_skew, zorder=5)

    # Helper Lines (Adiabats)
    z_range = np.linspace(0, 8, 50) * units.km
    p_range = mpcalc.height_to_pressure_std(z_range)

    # Dry Adiabats
    for theta in range(-20, 120, 10):
        theta_val = (theta + 273.15) * units.K
        t_adiabat = mpcalc.dry_lapse(p_range, theta_val, 1000 * units.hPa).to(units.degC).m
        ax1.plot(t_adiabat, z_range.m, color='brown', alpha=0.2, linewidth=1, transform=t_skew, zorder=2)

    # Mixing Ratio
    for w in [1, 2, 4, 7, 10, 16, 24]:
        w_val = w * units('g/kg')
        t_w = mpcalc.dewpoint_from_mixing_ratio(p_range, w_val).to(units.degC).m
        ax1.plot(t_w, z_range.m, color='green', alpha=0.15, linestyle=':', transform=t_skew, zorder=2)

    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.2)
    ax1.legend(loc='upper left')

    # --- PANEL 2: WIND SPEED ---
    ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='blue', alpha=0.1)
    ax2.set_ylim(0, 8)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.set_yticklabels([]) 
    ax2.grid(True, alpha=0.2)

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Paragliding Profile (Linear Z) | Payerne | {ref_time_str} UTC", fontsize=16)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Success: Skewed plot generated.")

if __name__ == "__main__":
    main()
