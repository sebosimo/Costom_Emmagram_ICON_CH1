import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob, pytz
import numpy as np

CACHE_DIR = "cache_data"

def main():
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)
    
    # 1. Extract values
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    u_ms, v_ms = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    # --- MOISTURE CALCULATION ---
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        # Relative Humidity - ensure it's 0-1 for MetPy
        rh = np.clip(ds["HUM"].values / 100.0, 0, 1.0)
        td = mpcalc.dewpoint_from_relative_humidity(t, rh)
    else:
        # Specific Humidity
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))
    
    # Physical fix: Dewpoint cannot exceed Temperature (ensures lines touch in clouds)
    td = np.minimum(td, t)

    z = mpcalc.pressure_to_height_std(p).to(units.km)
    u_kmh, v_kmh = u_ms.to('km/h').m, v_ms.to('km/h').m
    wind_speed_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

    inds = z.argsort()
    z_plot, t_plot, td_plot = z[inds].m, t[inds].m, td[inds].m
    u_plot, v_plot, wind_plot = u_kmh[inds], v_kmh[inds], wind_speed_kmh[inds]
    
    z_max = 7.0
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot = z_plot[mask], t_plot[mask], td_plot[mask]
    u_plot, v_plot, wind_plot = u_plot[mask], v_plot[mask], wind_plot[mask]

    # --- SKEW CONFIGURATION (Keep Original) ---
    SKEW_FACTOR = 5 
    def skew_x(temp, height): return temp + (height * SKEW_FACTOR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), sharey=True, 
                                   gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})
    
    ax1.set_ylim(0, z_max)
    skew_t, skew_td = skew_x(t_plot, z_plot), skew_x(td_plot, z_plot)
    min_x, max_x = min(np.min(skew_t), np.min(skew_td)), max(np.max(skew_t), np.max(skew_td))
    padding = 5
    ax1.set_xlim(min_x - padding, max_x + padding)

    # --- DRAW HELPER LINES (Keep Original) ---
    z_ref = np.linspace(0, z_max, 100) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)
    ax1.grid(True, axis='y', color='gray', alpha=0.3, linestyle='-', linewidth=0.8)

    for temp_base in range(-150, 151, 5):
        xb, xt = skew_x(temp_base, 0), skew_x(temp_base, z_max)
        if max(xb, xt) >= (min_x-padding) and min(xb, xt) <= (max_x+padding):
            ax1.plot([xb, xt], [0, z_max], color='blue', alpha=0.08, zorder=1)

    # Dry Adiabats
    for theta in range(-150, 301, 5):
        t_adiabat = mpcalc.dry_lapse(p_ref, (theta + 273.15) * units.K, 1000 * units.hPa).to(units.degC).m
        x_adiabat = skew_x(t_adiabat, z_ref.m)
        if np.max(x_adiabat) >= (min_x-padding) and np.min(x_adiabat) <= (max_x+padding):
            ax1.plot(x_adiabat, z_ref.m, color='brown', alpha=0.18, linewidth=1.2, zorder=2)

    # Mixing Ratio
    for t_start in range(-60, 61, 5):
        w_sat = mpcalc.mixing_ratio_from_relative_humidity(1000 * units.hPa, t_start * units.degC, 100 * units.percent)
        e_w = mpcalc.vapor_pressure(p_ref, w_sat)
        t_w = mpcalc.dewpoint(e_w).to(units.degC).m
        x_w = skew_x(t_w, z_ref.m)
        if np.max(x_w) >= (min_x-padding) and np.min(x_w) <= (max_x+padding):
            ax1.plot(x_w, z_ref.m, color='green', alpha=0.12, linestyle=':', zorder=2)

    # --- PLOT THERMO DATA (Keep Original Color Coding) ---
    dt, dz = np.diff(t_plot), np.diff(z_plot)
    lapse_rate = - (dt / np.where(dz==0, 0.001, dz)) 
    points = np.array([skew_t, z_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(vmin=-3, vmax=10), linewidth=4, zorder=5)
    lc.set_array(lapse_rate)
    ax1.add_collection(lc)
    ax1.plot(skew_td, z_plot, color='blue', linewidth=1, zorder=5, alpha=0.8)

    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)

    # --- PANEL 2: WIND ---
    ax2.plot(wind_plot, z_plot, color='blue', linewidth=2)
    ax2.set_xlim(0, 80) 
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    step = max(1, len(z_plot) // 14) 
    ax2.barbs(np.ones_like(z_plot[::step]) * 72, z_plot[::step], u_plot[::step], v_plot[::step], length=6)
    ax2.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- TIMEZONE & TITLE FIX ---
    utc_tz = pytz.utc
    ch_tz = pytz.timezone('Europe/Zurich')
    
    ref_utc = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).replace(tzinfo=utc_tz)
    # Horizon P0DT0H = 0 hours
    valid_utc = ref_utc + datetime.timedelta(hours=0) 
    
    ref_local = ref_utc.astimezone(ch_tz)
    valid_local = valid_utc.astimezone(ch_tz)

    title_line1 = f"Payerne | ICON-CH1 Run: {ref_utc.strftime('%H:%M')} UTC ({ref_local.strftime('%H:%M')} Local)"
    title_line2 = f"Valid: {valid_utc.strftime('%Y-%m-%d %H:%M')} UTC ({valid_local.strftime('%H:%M')} Local)"
    
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=14, y=0.97)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main()
