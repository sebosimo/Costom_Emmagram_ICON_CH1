import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import metpy.calc as mpcalc
from metpy.units import units
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
    p_hpa = (ds["P"].values * units.Pa).to(units.hPa)
    t_c = (ds["T"].values * units.K).to(units.degC)
    u, v = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        td_c = mpcalc.dewpoint_from_relative_humidity(t_c, ds["HUM"].values / 100.0)
    else:
        td_c = mpcalc.dewpoint_from_specific_humidity(p_hpa, t_c, ds["HUM"].values * units('kg/kg'))

    # Calculate wind speed in km/h
    wind_speed_kmh = mpcalc.wind_speed(u, v).to('km/h')

    # Sort Surface -> Space
    inds = p_hpa.argsort()[::-1]
    p, t, td, ws = p_hpa[inds].m, t_c[inds].m, td_c[inds].m, wind_speed_kmh[inds].m

    # --- 3. Figure Setup ---
    # We use a specific tall aspect ratio
    fig = plt.figure(figsize=(12, 14))
    
    # Panel 1: Emagram (Main) 
    # Panel 2: Wind (Right) - Starts exactly where Panel 1 ends (0.7) to close the gap
    ax1_rect = [0.1, 0.08, 0.6, 0.85]
    ax2_rect = [0.7, 0.08, 0.15, 0.85]
    
    ax1 = fig.add_axes(ax1_rect)
    ax2 = fig.add_axes(ax2_rect)

    # Global Limits
    P_BOT, P_TOP = 1020, 400
    T_LEFT, T_RIGHT = -40, 45

    # --- 4. PANEL 1: CUSTOM EMAGRAM ---
    ax1.set_yscale('log')
    ax1.set_ylim(P_BOT, P_TOP)
    ax1.set_xlim(T_LEFT, T_RIGHT)

    # A. Isoterms (Vertical lines)
    for iso_t in range(-60, 61, 10):
        ax1.axvline(iso_t, color='black', alpha=0.1, linewidth=0.8)

    # B. Dry Adiabats (Approx 1°C per 100m)
    # Using a reference atmosphere to calculate paths
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588
    for t_start in range(-20, 101, 10):
        t_adiabat = t_start - (0.0098 * z_ref) # Dry lapse rate
        ax1.plot(t_adiabat, p_ref, color='orangered', alpha=0.15, linewidth=0.8)

    # C. Mixing Ratio lines (g/kg)
    # Using your previously defined mixing ratio logic
    for w in [2, 5, 10, 20]:
        e = (w * p_ref) / (622 + w)
        ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
        t_m = (243.5 * ln_e) / (17.67 - ln_e)
        ax1.plot(t_m, p_ref, color='green', alpha=0.15, linestyle='--', linewidth=0.8)

    # Plot Actual Sounding
    ax1.plot(t, p, 'red', linewidth=3, label='Temperature')
    ax1.plot(td, p, 'green', linewidth=3, label='Dewpoint')

    # --- 5. PANEL 2: WIND SPEED ---
    ax2.set_yscale('log')
    ax2.set_ylim(P_BOT, P_TOP)
    ax2.set_xlim(0, 80)
    
    ax2.plot(ws, p, 'blue', linewidth=2)
    ax2.fill_betweenx(p, 0, ws, color='blue', alpha=0.1)

    # --- 6. SHARED ALTITUDE LABELS & HELPER LINES ---
    km_levels = np.arange(0, 9, 0.5)
    p_levels = mpcalc.height_to_pressure_std(km_levels * units.km).to(units.hPa).m
    
    for ax in [ax1, ax2]:
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.yaxis.set_minor_locator(NullLocator())
        # Horizontal lines extending across BOTH plots
        ax.grid(True, which='major', axis='y', color='black', alpha=0.2, linestyle='-')
        # Remove default pressure labels/ticks
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)

    # Custom KM label formatting on the far left only
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax1.tick_params(axis='y', labelleft=True, left=True)

    # --- 7. FINAL STYLING ---
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax2.set_xlabel("km/h", fontsize=12)
    ax2.set_xticks([0, 20, 40, 60, 80])
    ax1.legend(loc='upper left', frameon=True)

    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    fig.suptitle(f"Pilot Sounding Payerne | {ref_time_str} UTC", fontsize=16, y=0.96)

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Success: Cleaned Emagram generated.")

if __name__ == "__main__":
    main()
