import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import FixedLocator, MultipleLocator
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

    # Calculate Altitude (Z) linearly from pressure
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort by altitude (Surface up)
    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds], t[inds], td[inds], wind_speed[inds]
    p_plot = p[inds]

    # 3. Figure Setup
    fig = plt.figure(figsize=(14, 12))
    
    # --- Transformation Logic for 45 Degree Isotherms ---
    # We want a 1 degree C change to equal the visual distance of a 1 km change 
    # (or a specific ratio) to maintain the 45-degree angle.
    ax1 = fig.add_axes([0.1, 0.1, 0.55, 0.8])
    
    # Create the Skew Transformation
    # The '30' here is a scale factor to align the degree-to-km aspect ratio
    skew_angle = 45 
    base_trans = ax1.transData
    skew = transforms.Affine2D().skew_deg(skew_angle, 0)
    t_skew = skew + base_trans

    # --- PANEL 1: CUSTOM SKEW-T (LINEAR Z) ---
    ax1.set_ylim(0, 8)  # Linear KM
    ax1.set_xlim(-20, 40) # Temperature range
    
    # Plot T and Td using the skewed transformation
    ax1.plot(t_plot, z_plot, 'red', linewidth=2.5, label='Temp', transform=t_skew)
    ax1.plot(td_plot, z_plot, 'green', linewidth=2.5, label='Dewpoint', transform=t_skew)

    # Add Helper Lines (Adiabats & Mixing Ratio)
    # Define ranges for helper lines
    t_range = np.linspace(-60, 60, 100) * units.degC
    z_range = np.linspace(0, 8, 50) * units.km
    p_range = mpcalc.height_to_pressure_std(z_range)

    # Draw Isotherms (45 degrees)
    for temp in range(-50, 51, 10):
        ax1.axvline(temp, color='blue', alpha=0.15, linestyle='--', transform=t_skew)

    # Dry Adiabats
    for theta in range(-20, 100, 10):
        theta_val = (theta + 273.15) * units.K
        # Calculate T at each pressure level for this potential temperature
        t_adiabat = mpcalc.dry_lapse(p_range, theta_val, 1000 * units.hPa).to(units.degC)
        ax1.plot(t_adiabat, z_range, color='brown', alpha=0.2, linewidth=1, transform=t_skew)

    # Mixing Ratio Lines
    for w in [1, 2, 4, 7, 10, 16, 24]:
        w_val = w * units('g/kg')
        t_w = mpcalc.dewpoint_from_mixing_ratio(p_range, w_val).to(units.degC)
        ax1.plot(t_w, z_range, color='green', alpha=0.15, linestyle=':', transform=t_skew)

    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.legend(loc='upper right')

    # --- PANEL 2: WIND SPEED (LINEAR Z) ---
    ax2 = fig.add_axes([0.7, 0.1, 0.2, 0.8])
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='blue', alpha=0.1)
    
    ax2.set_ylim(0, 8)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("Wind Speed (km/h)", fontsize=12)
    ax2.set_yticklabels([]) # Sync with ax1
    ax2.grid(True, alpha=0.3)

    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Paragliding Profile (Linear Z) | Payerne | {ref_time_str} UTC", fontsize=16)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Generated linear altitude diagram with 45° skewed isotherms.")

if __name__ == "__main__":
    main()
