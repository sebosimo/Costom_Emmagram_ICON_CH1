import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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
    u_ms, v_ms = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    # Determine Dewpoint
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # --- CORRECTED THERMODYNAMICS ---
    # 1. Calculate vapor pressure from dewpoint
    e = mpcalc.vapor_pressure(p, td)
    # 2. Calculate mixing ratio from vapor pressure
    w = mpcalc.mixing_ratio(e, p)
    # 3. Calculate Virtual Temperature
    t_v = mpcalc.virtual_temperature(t, w).to(units.degC)

    z = mpcalc.pressure_to_height_std(p).to(units.km)
    wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

    inds = z.argsort()
    z_plot = z[inds].m
    t_plot, td_plot, tv_plot = t[inds].m, td[inds].m, t_v[inds].m
    wind_plot = wind_speed_kmh[inds]
    u_plot, v_plot = u_ms.to('km/h').m[inds], v_ms.to('km/h').m[inds]
    
    z_max = 7.0
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot, tv_plot, wind_plot = z_plot[mask], t_plot[mask], td_plot[mask], tv_plot[mask], wind_plot[mask]
    u_plot, v_plot = u_plot[mask], v_plot[mask]

    # --- SKEW CONFIGURATION ---
    SKEW_FACTOR = 5 
    def skew_x(temp, height):
        return temp + (height * SKEW_FACTOR)

    # 3. Figure Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharey=True, 
                                   gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})
    
    ax1.set_ylim(0, z_max)
    
    # --- DYNAMIC X-AXIS RANGE ---
    skew_t = skew_x(t_plot, z_plot)
    skew_td = skew_x(td_plot, z_plot)
    skew_tv = skew_x(tv_plot, z_plot)
    min_x = min(np.min(skew_t), np.min(skew_td), np.min(skew_tv))
    max_x = max(np.max(skew_t), np.max(skew_td), np.max(skew_tv))
    padding = 5
    ax1.set_xlim(min_x - padding, max_x + padding)

    # --- DRAW HELPER LINES ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)
    ax1.grid(True, axis='y', color='gray', alpha=0.3, linestyle='-', linewidth=0.8)

    # 1. Isotherms (Blue) - Every 5°C
    for temp_base in range(-150, 151, 5):
        xb, xt = skew_x(temp_base, 0), skew_x(temp_base, z_max)
        if max(xb, xt) >= (min_x-padding) and min(xb, xt) <= (max_x+padding):
            ax1.plot([xb, xt], [0, z_max], color='blue', alpha=0.08, zorder=1)

    # 2. Thermal Gradients (Orange Dashed 5°C/km)
    for temp_base in range(-150, 151, 5):
        t_grad_top = temp_base - (5.0 * z_max)
        x_grad_top = skew_x(t_grad_top, z_max)
        x_grad_bottom = skew_x(temp_base, 0)
        if max(x_grad_bottom, x_grad_top) >= (min_x-padding) and min(x_grad_bottom, x_grad_top) <= (max_x+padding):
            ax1.plot([x_grad_bottom, x_grad_top], [0, z_max], color='orange', linestyle='--', alpha=0.3)

    # 3. Dry Adiabats (Brown)
    for theta in range(-150, 301, 5):
        t_adiabat = mpcalc.dry_lapse(p_ref, (theta + 273.15) * units.K, 1000 * units.hPa).to(units.degC).m
        x_adiabat = skew_x(t_adiabat, z_ref.m)
        if np.max(x_adiabat) >= (min_x-padding) and np.min(x_adiabat) <= (max_x+padding):
            ax1.plot(x_adiabat, z_ref.m, color='brown', alpha=0.18, linewidth=1.2)

    # 4. Mixing Ratio Lines (Green) - Every 5°C
    for t_start in range(-60, 61, 5):
        w_sat = mpcalc.mixing_ratio_from_relative_humidity(1000 * units.hPa, t_start * units.degC, 100 * units.percent)
        e_w = mpcalc.vapor_pressure(p_ref, w_sat)
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        x_w = skew_x(t_w, z_ref.m)
        if np.max(x_w) >= (min_x-padding) and np.min(x_w) <= (max_x+padding):
            ax1.plot(x_w, z_ref.m, color='green', alpha=0.12, linestyle=':')

    # --- PLOT THERMO DATA ---
    dt, dz = np.diff(t_plot), np.diff(z_plot)
    lapse_rate = - (dt / dz) 
    points = np.array([skew_t, z_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm, cmap = Normalize(vmin=-3, vmax=10), plt.get_cmap('RdYlGn') 
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4, zorder=5)
    lc.set_array(lapse_rate)
    ax1.add_collection(lc)

    # Traces
    ax1.plot(skew_td, z_plot, color='blue', linewidth=1, zorder=5, alpha=0.8, label='Dewpoint')
    # Virtual Temp Trace
    ax1.plot(skew_tv, z_plot, color='orange', linewidth=1.5, linestyle='--', zorder=6, label='Virtual Temp')

    # Formatting
    visible_ticks = [t for t in np.arange(-150, 151, 5) if (min_x-padding) <= t <= (max_x+padding)]
    ax1.set_xticks(visible_ticks)
    ax1.set_xticklabels(visible_ticks)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.legend(loc='upper left', frameon=True)

    # Wind Panel
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.set_xlim(0, 80) 
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.grid(True, axis='both', color='gray', alpha=0.2)
    ax2.yaxis.set_tick_params(which='both', left=False, right=False)

    ref_dt = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
    fig.suptitle(f"Payerne Sounding | {ref_dt.strftime('%Y-%m-%d %H:%M')} UTC\nVirtual Temp (Orange) | 5°C Mixing Lines (Green)", fontsize=14, y=0.96)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
