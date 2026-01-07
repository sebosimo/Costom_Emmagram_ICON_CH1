import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os, glob, datetime
from matplotlib.ticker import FixedLocator, FuncFormatter

CACHE_DIR = "cache_data"

def main():
    # 1. Load Data
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files: return
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)

    # 2. Extract Data from GRIB/NetCDF source
    p_hpa = ds["P"].values / 100.0  
    t_c = ds["T"].values - 273.15 
    u, v = ds["U"].values, ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 

    # --- CORE FIX: USE MEASURED HUMIDITY DATA ---
    # We use the Magnus-Tetens formula on the actual humidity variable
    rel_hum = ds["HUM"].values # Measured RH from GRIB 
    # es: saturation vapor pressure
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    # e: actual vapor pressure derived from measured RH
    e = (rel_hum / 100.0) * es
    # td: dewpoint temperature calculated from actual vapor pressure
    ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
    td_c = (243.5 * ln_e) / (17.67 - ln_e)

    # Sort data Surface-Upwards for consistent plotting
    idx = p_hpa.argsort()[::-1]
    p, t, td, ws = p_hpa[idx], t_c[idx], td_c[idx], ws_kmh[idx]

    # --- SKEW CONFIGURATION (MeteoSwiss Style) ---
    SKEW_FACTOR = 48 
    P_BOT, P_TOP = 1020, 400

    def get_skew_x(temp, press):
        return temp + SKEW_FACTOR * np.log10(1020 / press)

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(10, 14))
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    ax2 = fig.add_axes([0.75, 0.1, 0.15, 0.8])

    # 3. BACKGROUND GRID (Manual Lines)
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # Isoterms (Vertical skewed lines)
    for iso_t in range(-80, 60, 10):
        ax1.plot(get_skew_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, lw=0.6)

    # Dry Adiabats (1°C per 100m)
    for t_start in range(-20, 100, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(get_skew_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, lw=0.8)

    # Mixing Ratio (g/kg)
    for w in [1, 2, 5, 10, 20]:
        ew = (w * p_ref) / (622 + w)
        ln_ew = np.log(np.clip(ew / 6.112, 1e-5, None))
        tw = (243.5 * ln_ew) / (17.67 - ln_ew)
        ax1.plot(get_skew_x(tw, p_ref), p_ref, color='green', alpha=0.15, ls='--', lw=0.8)

    # 4. PLOT MEASURED DATA (Temperature and Dewpoint)
    ax1.plot(get_skew_x(t, p), p, color='red', linewidth=3, label='Temperature')
    ax1.plot(get_skew_x(td, p), p, color='green', linewidth=2, linestyle='--', label='Dewpoint')

    # 5. WIND PROFILE (km/h)
    ax2.plot(ws, p, color='red', linewidth=2)
    ax2.fill_betweenx(p, 0, ws, color='red', alpha=0.05)
    ax2.set_xlim(0, 100)

    # 6. ALTITUDE GRID & Hectopascal Removal
    km_all = np.arange(0, 8.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_all * 1000))**5.25588

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1)

    # Custom KM Labels on primary axis
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])}" if km_all[pos] % 1 == 0 else ""))
    ax1.tick_params(axis='y', labelleft=True, left=True)

    # 7. FINAL LAYOUT
    # Expand left limits to -80 for dry dewpoint visibility
    ax1.set_xlim(get_skew_x(-80, P_BOT), get_skew_x(45, P_BOT))
    ax1.set_ylabel("km", rotation=0, loc='top', labelpad=-20)
    ax1.set_xlabel("°C")
    ax2.set_xlabel("km/h")
    
    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Sounding Payerne | {ref_time} UTC", fontsize=16, y=0.94)
    ax1.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Accurate emagram generated using GRIB moisture data.")

if __name__ == "__main__":
    main()
