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
    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K).to(units.degC)
    u, v = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    if ds.attrs.get("HUM_TYPE") == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # --- CORE THERMODYNAMIC FIXES ---
    # 1. Partial vapor pressure (e) from dewpoint
    e = mpcalc.saturation_vapor_pressure(td)
    # 2. Mixing ratio (w) from partial pressure
    w = mpcalc.mixing_ratio(e, p)
    # 3. Virtual Temperature (Tv)
    t_v = mpcalc.virtual_temperature(t, w).to(units.degC)
    # 4. Wind speed in km/h
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')

    # Sort Surface -> Space
    inds = p.argsort()[::-1]
    p_hpa, t_plot, td_plot, tv_plot, ws_plot = p[inds].to(units.hPa).m, t[inds].m, td[inds].m, t_v[inds].m, wind_speed[inds].m

    # --- 3. SKEW CONFIGURATION ---
    # SKEW_FACTOR 48 approximates a 45-degree angle on a log-P chart
    SKEW_FACTOR = 48 
    P_BOT, P_TOP = 1020, 400

    def get_skew_x(temp, press):
        """Calculates skewed x-coordinate: X = T + C * log(P_surface / P)"""
        return temp + SKEW_FACTOR * np.log10(1020 / press)

    # --- 4. FIGURE SETUP ---
    fig = plt.figure(figsize=(12, 16))
    # Panel 1: Emagram (0.1 to 0.7), Panel 2: Wind (0.7 to 0.88) -> No gap
    ax1_box = [0.1, 0.1, 0.6, 0.8]
    ax2_box = [0.7, 0.1, 0.18, 0.8]
    
    ax1 = fig.add_axes(ax1_box)
    ax2 = fig.add_axes(ax2_box)

    # 5. BACKGROUND PHYSICS GRID
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # A. Slanted Isoterms (Solid gray)
    for iso_t in range(-80, 81, 10):
        ax1.plot(get_skew_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, lw=0.6)

    # B. Dry Adiabats (1°C / 100m)
    for t_start in range(-40, 101, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(get_skew_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, lw=0.8)

    # C. Mixing Ratio lines EVERY 5°C
    # We plot the isohume corresponding to saturation at 1000hPa for every 5C
    for t_w in range(-40, 41, 5):
        w_val = mpcalc.mixing_ratio_from_relative_humidity(1000 * units.hPa, t_w * units.degC, 100 * units.percent)
        e_path = mpcalc.vapor_pressure(p_ref * units.hPa, w_val)
        td_path = mpcalc.dewpoint(e_path).m
        ax1.plot(get_skew_x(td_path, p_ref), p_ref, color='green', alpha=0.12, ls='--', lw=0.7)

    # 6. PLOT ACTUAL DATA (Skewed)
    ax1.plot(get_skew_x(t_plot, p_hpa), p_hpa, color='red', lw=3.5, label='Temperature')
    ax1.plot(get_skew_x(td_plot, p_hpa), p_hpa, color='green', lw=2.5, ls='--', label='Dewpoint')
    ax1.plot(get_skew_x(tv_plot, p_hpa), p_hpa, color='orange', lw=1.5, ls=':', label='Virtual Temp')

    # 7. WIND PROFILE
    ax2.plot(ws_plot, p_hpa, color='blue', lw=2)
    ax2.fill_betweenx(p_hpa, 0, ws_plot, color='blue', alpha=0.05)
    ax2.set_xlim(0, 100)

    # 8. SHARED ALTITUDE & CLEANUP
    km_levels = np.arange(0, 8.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_levels * 1000))**5.25588
    
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.yaxis.set_minor_locator(NullLocator())
        # Horizontal lines extending across BOTH plots
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1, ls='-')
        # Remove all hPa text and standard ticks
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)

    # KM labels only on far left
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax1.tick_params(axis='y', labelleft=True, left=True)

    # Final Layout
    ax1.set_xlim(get_skew_x(-50, P_BOT), get_skew_x(40, P_BOT))
    ax1.set_xlabel("Temperature (°C)")
    ax2.set_xlabel("km/h")
    ax1.legend(loc='upper left', frameon=True)

    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Pilot Sounding Payerne | {ref_time} UTC", fontsize=16, y=0.95)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Corrected and Skewed Emagram generated.")

if __name__ == "__main__":
    main()
