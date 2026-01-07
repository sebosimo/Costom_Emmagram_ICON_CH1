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

    # 2. Extract Data & Units
    p_hpa = ds["P"].values / 100.0  
    t_c = ds["T"].values - 273.15 
    u, v = ds["U"].values, ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 

    # Sort Surface Upwards
    idx = p_hpa.argsort()[::-1]
    p, t, ws = p_hpa[idx], t_c[idx], ws_kmh[idx]

    # Dewpoint (Pilot approx)
    hum = ds["HUM"].values[idx]
    td = t - ((100 - hum) / 5)

    # --- SKEW CONFIGURATION ---
    SKEW = 45 # The "Lean"
    P_BOT, P_TOP = 1020, 400

    def skew_x(temp, press):
        # The core transformation for an Emagram/Skew-T
        return temp + SKEW * np.log10(P_BOT / press)

    # 3. Figure Setup (Manual axis placement for tall aspect)
    fig = plt.figure(figsize=(10, 14))
    # Emagram box: [left, bottom, width, height]
    ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    # Wind box: attached directly with no gap
    ax2 = fig.add_axes([0.7, 0.1, 0.15, 0.8])

    # 4. BACKGROUND GRID (Manual Lines)
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # A. Isoterms (Vertical lines in Temperature space, skewed here)
    for iso_t in range(-60, 60, 10):
        ax1.plot(skew_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, lw=0.5)

    # B. Dry Adiabats (1°C / 100m)
    for t_start in range(-20, 80, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(skew_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, lw=0.7)

    # C. Mixing Ratio (g/kg)
    for w in [2, 5, 10, 20]:
        e = (w * p_ref) / (622 + w)
        ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
        t_m = (243.5 * ln_e) / (17.67 - ln_e)
        ax1.plot(skew_x(t_m, p_ref), p_ref, color='green', alpha=0.15, ls='--', lw=0.7)

    # 5. PLOT SOUNDING
    ax1.plot(skew_x(t, p), p, color='red', lw=3, label='Temp')
    ax1.plot(skew_x(td, p), p, color='green', lw=2, ls='--', label='Dewp')

    # 6. PANEL 2: WIND SPEED
    ax2.plot(ws, p, color='blue', lw=2)
    ax2.fill_betweenx(p, 0, ws, color='blue', alpha=0.1)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("km/h", fontsize=10)

    # 7. ALTITUDE & COORDINATE SYNC
    km_levels = np.arange(0, 8.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_levels * 1000))**5.25588

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1, ls='-')

    # Formatting
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax2.set_yticklabels([]) # No pressure labels on wind plot
    
    # Range management
    ax1.set_xlim(skew_x(-30, P_BOT), skew_x(40, P_BOT))
    ax1.set_ylabel("Altitude", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    
    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Paraglider Sounding Payerne | {ref_time} UTC", fontsize=14, y=0.95)
    ax1.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Pure-Matplotlib tall sounding generated.")

if __name__ == "__main__":
    main()
