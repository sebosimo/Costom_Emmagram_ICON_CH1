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

    # 2. Extract Data
    p_hpa = ds["P"].values / 100.0  
    t_c = ds["T"].values - 273.15 
    u, v = ds["U"].values, ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 

    # Sort Surface Upwards
    idx = p_hpa.argsort()[::-1]
    p, t, ws = p_hpa[idx], t_c[idx], ws_kmh[idx]

    # Calculate Dewpoint (Td)
    hum = ds["HUM"].values[idx]
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        td = t - ((100 - hum) / 5) # Standard glider pilot approximation
    else:
        td = t 

    # --- SKEW CONFIGURATION ---
    # SKEW_FACTOR controls the 'lean' to the right. 40-50 matches MeteoSwiss.
    SKEW_FACTOR = 48 
    P_BOT, P_TOP = 1020, 400

    def get_x(temp, press):
        """Applies the skew transformation: x = T + skew * log(P_surface / P)"""
        return temp + SKEW_FACTOR * np.log10(1020 / press)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 16))
    # Panel 1: Emagram, Panel 2: Wind (0.05 spacing for 'no gap' feel)
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    ax2 = fig.add_axes([0.75, 0.1, 0.18, 0.8])

    # 3. BACKGROUND GRID
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # A. Isoterms (Solid Vertical-ish lines in skewed space)
    for iso_t in range(-60, 60, 10):
        ax1.plot(get_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, linewidth=0.6)

    # B. Dry Adiabats (1°C per 100m)
    for t_start in range(-20, 80, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(get_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, linewidth=0.8)

    # C. Mixing Ratio (g/kg)
    for w in [2, 5, 10, 20]:
        e = (w * p_ref) / (622 + w)
        ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
        t_m = (243.5 * ln_e) / (17.67 - ln_e)
        ax1.plot(get_x(t_m, p_ref), p_ref, color='green', alpha=0.15, linestyle='--', linewidth=0.8)

    # 4. PLOT SOUNDING (Correctly skewed)
    ax1.plot(get_x(t, p), p, color='red', linewidth=3.5, label='Temperature')
    ax1.plot(get_x(td, p), p, color='green', linewidth=2.5, linestyle='--', label='Dewpoint')

    # 5. PANEL 2: WIND SPEED
    ax2.plot(ws, p, color='blue', linewidth=2)
    ax2.fill_betweenx(p, 0, ws, color='blue', alpha=0.08)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("km/h", fontsize=11)

    # 6. ALTITUDE & COORDINATE SYNC
    km_levels = np.arange(0, 8.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_levels * 1000))**5.25588

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        # Extended helper lines across both panels
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1, linestyle='-')

    # Force KM labels on ax1, remove labels on ax2
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax2.set_yticklabels([])

    # 7. CLEANUP
    # Set X-Limits to show the relevant data area
    ax1.set_xlim(get_x(-35, 1020), get_x(35, 1020))
    ax1.set_ylabel("Altitude", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    
    # Title and Legend
    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    fig.suptitle(f"Pilot Sounding Payerne | {ref_time} UTC", fontsize=16, y=0.94)
    ax1.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: MeteoSwiss-style Custom Emagram generated.")

if __name__ == "__main__":
    main()
