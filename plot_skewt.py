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
    u_ms, v_ms = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    z = mpcalc.pressure_to_height_std(p).to(units.km)
    
    # Scientific conversion to km/h
    u_kmh = u_ms.to('km/h').m
    v_kmh = v_ms.to('km/h').m
    wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

    inds = z.argsort()
    z_plot = z[inds].m
    t_plot, td_plot = t[inds].m, td[inds].m
    u_plot, v_plot, wind_plot = u_kmh[inds], v_kmh[inds], wind_speed_kmh[inds]
    
    z_max = 7.0
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot, wind_plot = z_plot[mask], t_plot[mask], td_plot[mask], wind_plot[mask]
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
    min_x, max_x = min(np.min(skew_t), np.min(skew_td)), max(np.max(skew_t), np.max(skew_td))
    padding = 5
    ax1.set_xlim(min_x - padding, max_x + padding)

    # --- DRAW HELPER LINES ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)
    ax1.grid(True, axis='y', color='gray', alpha=0.3, linestyle='-', linewidth=0.8)

    # 1. Isotherms (Blue) and 2. Thermal Threshold (Orange Dashed 0.5C/100m)
    for temp_base in range(-150, 151, 5):
        # Isotherm coordinates
        xb, xt = skew_x(temp_base, 0), skew_x(temp_base, z_max)
        
        # Thermal Gradient coordinates (0.5C/100m = 5C/km)
        t_grad_top = temp_base - (5.0 * z_max)
        x_grad_top = skew_x(t_grad_top, z_max)
        x_grad_bottom = skew_x(temp_base, 0)

        # Visibility Check: Draw if either the Isotherm OR the Gradient line is in view
        if (max(xb, xt) >= (min_x-padding) and min(xb, xt) <= (max_x+padding)) or \
           (max(x_grad_bottom, x_grad_top) >= (min_x-padding) and min(x_grad_bottom, x_grad_top) <= (max_x+padding)):
            
            # Draw Isotherm
            ax1.plot([xb, xt], [0, z_max], color='blue', alpha=0.08, zorder=1)
            
            # Draw Thermal Gradient
            ax1.plot([x_grad_bottom, x_grad_top], [0, z_max], color='orange', 
                     linestyle='--', linewidth=1, alpha=0.18, zorder=1)

    # 3. Dry Adiabats (Brown)
    for theta in range(-150, 301, 5):
        t_adiabat = mpcalc.dry_lapse(p_ref, (theta + 273.15) * units.K, 1000 * units.hPa).to(units.degC).m
        x_adiabat = skew_x(t_adiabat, z_ref.m)
        if np.max(x_adiabat) >= (min_x-padding) and np.min(x_adiabat) <= (max_x+padding):
            ax1.plot(x_adiabat, z_ref.m, color='brown', alpha=0.18, linewidth=1.2, zorder=2)

    # 4. Mixing Ratio Lines (Green)
    for w in [0.5, 1, 2, 4, 7, 10, 16, 24, 32]:
        e_w = mpcalc.vapor_pressure(p_ref, w * units('g/kg'))
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        x_w = skew_x(t_w, z_ref.m)
        if np.max(x_w) >= (min_x-padding) and np.min(x_w) <= (max_x+padding):
            ax1.plot(x_w, z_ref.m, color='green', alpha=0.15, linestyle=':', zorder=2)

    # --- PLOT THERMO DATA ---
    ax1.plot(skew_t, z_plot, 'red', linewidth=3, zorder=5)
    ax1.plot(skew_td, z_plot, 'green', linewidth=3, zorder=5)

    visible_ticks = [t for t in np.arange(-150, 151, 10) if (min_x-padding) <= t <= (max_x+padding)]
    ax1.set_xticks(visible_ticks)
    ax1.set_xticklabels(visible_ticks)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)

    # --- PANEL 2: WIND SPEED & BARBS ---
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.set_xlim(0, 80) 
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.set_xticks(np.arange(0, 81, 10))
    ax2.grid(True, axis='both', color='gray', alpha=0.2)
    ax2.yaxis.set_tick_params(which='both', left=False, right=False)

    step = max(1, len(z_plot) // 14) 
    ax2.barbs(np.ones_like(z_plot[::step]) * 72, z_plot[::step], 
              u_plot[::step], v_plot[::step], 
              barb_increments=dict(half=5, full=10, flag=50),
              length=6, color='black', alpha=0.8)

    ax2.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- REVERTED TITLE FORMAT ---
    ref_dt = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
    output_dt = ref_dt
    lead_hours = int((output_dt - ref_dt).total_seconds() // 3600)

    title_str = (f"Payerne | ICON-CH1 Run: {ref_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
                 f"Output: {output_dt.strftime('%Y-%m-%d %H:%M')} UTC (+{lead_hours}h)")
    fig.suptitle(title_str, fontsize=14, y=0.96)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
