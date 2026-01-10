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
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)
    
    # 1. Physical Data Extraction
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    
    # Accurate Dewpoint Calculation
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        # Ensure RH is a fraction (0-1) for MetPy
        rh_val = np.clip(ds["HUM"].values / 100.0, 0, 1.0)
        td = mpcalc.dewpoint_from_relative_humidity(t, rh_val)
    else:
        qv = ds["HUM"].values * units('kg/kg')
        td = mpcalc.dewpoint_from_specific_humidity(p, t, qv)

    # Physical Constraint: Dewpoint cannot exceed Temperature
    td = np.minimum(td, t)

    u_ms = ds["U"].values * units('m/s')
    v_ms = ds["V"].values * units('m/s')
    z = mpcalc.pressure_to_height_std(p).to(units.km)
    
    # 2. Sorting and Masking
    inds = z.argsort()
    z_plot = z[inds].m
    t_plot, td_plot = t[inds].m, td[inds].m
    u_kmh = u_ms[inds].to('km/h').m
    v_kmh = v_ms[inds].to('km/h').m
    wind_speed_kmh = mpcalc.wind_speed(u_ms[inds], v_ms[inds]).to('km/h').m

    z_max = 8.0 # Standard flight/sounding height for ICON-CH1 focus
    mask = z_plot <= z_max
    z_plot, t_plot, td_plot = z_plot[mask], t_plot[mask], td_plot[mask]
    u_plot, v_plot, wind_plot = u_kmh[mask], v_kmh[mask], wind_speed_kmh[mask]

    # --- SKEW CONFIGURATION ---
    # To make lines "touch" correctly, the skew transformation must be mathematically identical
    SKEW_FACTOR = 6.0 
    def skew_x(temp, height_km):
        return temp + (height_km * SKEW_FACTOR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True, 
                                   gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.02})
    
    ax1.set_ylim(0, z_max)
    skew_t = skew_x(t_plot, z_plot)
    skew_td = skew_x(td_plot, z_plot)
    
    # Auto-limit X axis based on data
    ax1.set_xlim(np.min(skew_td)-5, np.max(skew_t)+10)

    # --- DRAW BACKGROUND GRID ---
    z_ref = np.linspace(0, z_max, 50) * units.km
    p_ref = mpcalc.height_to_pressure_std(z_ref)

    # 1. Isotherms (Blue)
    for temp_base in range(-100, 60, 10):
        xb = skew_x(temp_base, 0)
        xt = skew_x(temp_base, z_max)
        ax1.plot([xb, xt], [0, z_max], color='blue', alpha=0.1, linewidth=0.8, zorder=1)

    # 2. Dry Adiabats (Brown)
    for theta in range(-60, 150, 10):
        # Calculate T along dry adiabat: T = Theta * (P/P0)^R/Cp
        t_adiabat = mpcalc.dry_lapse(p_ref, (theta + 273.15) * units.K, 1000 * units.hPa).to(units.degC).m
        ax1.plot(skew_x(t_adiabat, z_ref.m), z_ref.m, color='brown', alpha=0.15, linewidth=1, zorder=1)

    # --- PLOT DATA ---
    # Color-coded Lapse Rate Line
    dt = np.diff(t_plot)
    dz = np.diff(z_plot)
    # Avoid division by zero
    dz[dz == 0] = 0.001
    lapse_rate = - (dt / dz) 
    
    points = np.array([skew_t, z_plot]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(vmin=-2, vmax=10), linewidth=3.5, zorder=5)
    lc.set_array(lapse_rate)
    ax1.add_collection(lc)

    # Dewpoint Line (Blue)
    ax1.plot(skew_td, z_plot, color='blue', linewidth=2, alpha=0.8, zorder=6, label="Dewpoint")

    # Labels
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C) [Skew-T]", fontsize=12)
    ax1.grid(True, axis='y', alpha=0.2)

    # --- PANEL 2: WIND ---
    ax2.plot(wind_plot, z_plot, color='black', linewidth=1.5)
    ax2.fill_betweenx(z_plot, 0, wind_plot, color='purple', alpha=0.1)
    ax2.set_xlim(0, max(100, np.max(wind_plot)+10)) 
    ax2.set_xlabel("Wind Speed (km/h)", fontsize=12)
    
    # Barbs
    step = max(1, len(z_plot) // 15)
    ax2.barbs(np.full_like(z_plot[::step], ax2.get_xlim()[1]*0.8), z_plot[::step], 
              u_plot[::step], v_plot[::step], length=6, pivot='middle')

    # --- TITLE & TIMESTAMPS ---
    try:
        ref_dt = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
        valid_dt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
        lead_h = ds.attrs.get("lead_time", "0")
        
        title_str = (f"Payerne (ICON-CH1) | Run: {ref_dt.strftime('%Y-%m-%d %H:%M')} UTC\n"
                     f"Valid: {valid_dt.strftime('%Y-%m-%d %H:%M')} UTC (+{lead_h}h)")
    except:
        title_str = "ICON-CH1 Vertical Profile"

    fig.suptitle(title_str, fontsize=16, y=0.97)
    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Plot generated: latest_skewt.png")

if __name__ == "__main__":
    main()
