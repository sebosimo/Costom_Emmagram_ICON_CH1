import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import xarray as xr
import os, datetime, glob
import numpy as np

CACHE_DIR = "cache_data"

def main():
    # 1. Load Data
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files: return
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)
    
    # 2. Extract Measured Data
    p_hpa = ds["P"].values / 100.0  
    t_c = ds["T"].values - 273.15 
    rel_hum = ds["HUM"].values 
    u, v = ds["U"].values, ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 

    # --- RESTORED DEWPOINT LOGIC (Measured) ---
    # Using the Magnus-Tetens formula on the actual GRIB humidity data
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = (rel_hum / 100.0) * es
    ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
    td_c = (243.5 * ln_e) / (17.67 - ln_e)

    # Sort Surface -> Space
    idx = p_hpa.argsort()[::-1]
    p, t, td, ws = p_hpa[idx], t_c[idx], td_c[idx], ws_kmh[idx]

    # --- 3. SKEW-T TRANSFORMATION LOGIC ---
    SKEW = 48 # Matches MeteoSwiss lean
    P_BOT = 1020
    
    def skew_x(temp, press):
        """Transforms Temperature into Skewed X-coordinate based on log-P"""
        return temp + SKEW * np.log10(P_BOT / press)

    # --- 4. FIGURE SETUP ---
    fig = plt.figure(figsize=(12, 16))
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8]) 
    ax2 = fig.add_axes([0.75, 0.1, 0.15, 0.8]) 

    P_TOP = 400
    # Expanded limits to handle skewed cold dewpoints on the left
    T_MIN, T_MAX = -70, 45 

    # --- 5. BACKGROUND PHYSICS GRID ---
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # A. Slanted Isotherms (Vertical temperature reference)
    for iso_t in range(-80, 81, 10):
        ax1.plot(skew_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, lw=0.6)

    # B. Dry Adiabats (1°C / 100m)
    for t_start in range(-40, 121, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(skew_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, lw=0.8)

    # C. Mixing Ratio (g/kg - green dashed)
    for w in [1, 2, 5, 10, 20]:
        ew = (w * p_ref) / (622 + w)
        ln_ew = np.log(np.clip(ew / 6.112, 1e-5, None))
        tw = (243.5 * ln_ew) / (17.67 - ln_ew)
        ax1.plot(skew_x(tw, p_ref), p_ref, color='green', alpha=0.15, ls='--', lw=0.8)

    # --- 6. PLOT ACTUAL SOUNDING DATA ---
    ax1.plot(skew_x(t, p), p, color='red', lw=3.5, label='Temperature')
    ax1.plot(skew_x(td, p), p, color='green', lw=2.5, ls='--', label='Dewpoint')

    # --- 7. PANEL 2: WIND PROFILE ---
    ax2.plot(ws, p, color='blue', lw=2)
    ax2.fill_betweenx(p, 0, ws, color='blue', alpha=0.08)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("km/h")

    # --- 8. SHARED ALTITUDE & CLEANUP ---
    km_levels = np.arange(0, 9, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_levels * 1000))**5.25588
    
    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1, ls='-')
        ax.tick_params(axis='y', which='both', labelleft=False, left=False)

    # Labels only on far left
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax1.tick_params(axis='y', labelleft=True, left=True)

    # --- 9. FINAL TOUCHES ---
    ax1.set_xlim(skew_x(T_MIN, P_BOT), skew_x(T_MAX, P_BOT))
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("Altitude")
    ax1.legend(loc='upper left', frameon=True)

    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Pilot Sounding Payerne | {ref_time} UTC", fontsize=16, y=0.95)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Skewed sounding with restored dewpoint generated.")

if __name__ == "__main__":
    main()
