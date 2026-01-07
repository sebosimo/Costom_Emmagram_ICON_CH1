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

    # Calculate Standard Altitude (km) and Speed (km/h)
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort and filter for 0-7 km range
    inds = z.argsort()
    z_plot, t_plot, td_plot, wind_plot = z[inds].m, t[inds].m, td[inds].m, wind_speed[inds].m
    u_plot, v_plot = u[inds].m, v[inds].m 
    
    z_max = 7.0
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot, wind_plot = z_plot[mask], t_plot[mask], td_plot[mask], wind_plot[mask]
    u_plot, v_plot = u_plot[mask], v_plot[mask]

    # --- SKEW CONFIGURATION ---
    SKEW_FACTOR = 8 
    def skew_x(temp, height):
        return temp + (height * SKEW_FACTOR)

    # 3. Figure Setup
    # Total width 30 inches; Thermodynamic:Wind ratio at 6:1 for more wind room
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10), sharey=True, 
                                   gridspec_kw={'width_ratios': [6, 1], 'wspace': 0})
    
    ax1.set_ylim(0, z_max)
    
    # --- DYNAMIC X-AXIS RANGE ---
    # Calculates bounds based on actual data to remove empty space
    skew_t = skew_x(t_plot, z_plot)
    skew_td = skew_x(td_plot, z_plot)
    
    min_x = min(np.min(skew_t), np.min(skew_td))
    max_x = max(np.max(skew_t), np.max(skew_td))
    
    padding = 5
    left_bound = min_x - padding
    right_bound = max_x + padding
    ax1.set_xlim(left_bound, right_bound)

    # --- DRAW HELPER LINES ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)

    # Horizontal Altitude Lines
    ax1.grid(True, axis='y', color='gray', alpha=0.3, linestyle='-', linewidth=0.8)

    # Tilted Isotherms (Every 10°C)
    for temp in range(-100, 101, 10):
        x_bottom = skew_x(temp, 0)
        x_top = skew_x(temp, z_max)
        if max(x_bottom, x_top) >= left_bound and min(x_bottom, x_top) <= right_bound:
            ax1.plot([x_bottom, x_top], [0, z_max], color='blue', alpha=0.08, zorder=1)

    # Dry Adiabats
    for theta in range(-100, 251, 10):
        theta_val = (theta + 273.15) * units.K
        t_adiabat = mpcalc.dry_lapse(p_ref, theta_val, 1000 * units.hPa).to(units.degC).m
        x_adiabat = skew_x(t_adiabat, z_ref.m)
        if np.max(x_adiabat) >= left_bound and np.min(x_adiabat) <= right_bound:
            ax1.plot(x_adiabat, z_ref.m, color='brown', alpha=0.18, linewidth=1.2, zorder=2)

    # Mixing Ratio Lines
    for w in [0.5, 1, 2, 4, 7, 10, 16, 24, 32]:
        e_w = mpcalc.vapor_pressure(p_ref, w * units('g/kg'))
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        x_w = skew_x(t_w, z_ref.m)
        if np.max(x_w) >= left_bound and np.min(x_w) <= right_bound:
            ax1.plot(x_w, z_ref.m, color='green', alpha=0.15, linestyle=':', zorder=2)

    # --- PLOT THERMO DATA ---
    ax1.plot(skew_t, z_plot, 'red', linewidth=3, label='Temp', zorder=5)
    ax1.plot(skew_td, z_plot, 'green', linewidth=3, label='Dewpoint', zorder=5)

    # X-Axis Tick Labels referenced to surface temp
    temp_ticks = np.arange(-100, 101, 10)
    visible_ticks = [t for t in temp_ticks if left_bound <= t <= right_bound]
    ax1.set_xticks(visible_ticks)
    ax1.set_xticklabels(visible_ticks)
    
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.legend(loc='upper right', frameon=True)

    # --- PANEL 2: WIND SPEED & BARBS ---
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.set_xlim(0, 50) 
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.grid(True, axis='both', alpha=0.2)
    
    # Wind Barbs sampled every ~500m
    step = max(1, len(z_plot) // 14) 
    ax2.barbs(np.ones_like(z_plot[::step]) * 45, z_plot[::step], 
              u_plot[::step], v_plot[::step], length=6, color='black',
