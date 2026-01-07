import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import os, glob, datetime
from matplotlib.ticker import FixedLocator, FuncFormatter

CACHE_DIR = "cache_data"

def main():
    # 1. Load the most recent data file
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found.")
        return
    latest_file = max(files, key=os.path.getctime)
    ds = xr.open_dataset(latest_file)

    # 2. Extract and format data
    p_hpa = ds["P"].values / 100.0  
    t_c = ds["T"].values - 273.15 
    u, v = ds["U"].values, ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 

    # Sort data for ascending height
    idx = p_hpa.argsort()[::-1]
    p, t, ws = p_hpa[idx], t_c[idx], ws_kmh[idx]

    # Pilot approximation for Dewpoint (Td)
    hum = ds["HUM"].values[idx]
    td = t - ((100 - hum) / 5)

    # --- SKEW CONFIGURATION ---
    # SKEW_FACTOR 45-50 matches the sharp lean of MeteoSwiss charts
    SKEW_FACTOR = 48 
    P_BOT, P_TOP = 1020, 400

    def get_skew_x(temp, press):
        """Calculates skewed x-coordinate based on temperature and pressure."""
        return temp + SKEW_FACTOR * np.log10(1020 / press)

    # --- FIGURE SETUP ---
    fig = plt.figure(figsize=(10, 14))
    
    # Custom axes boxes: [left, bottom, width, height]
    ax1 = fig.add_axes([0.1, 0.1, 0.65, 0.8])
    ax2 = fig.add_axes([0.75, 0.1, 0.15, 0.8])

    # 3. BACKGROUND GRID (Manual Lines)
    # Define altitude range for lines (0 to 8km)
    z_ref = np.linspace(0, 8000, 100)
    p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588

    # A. Isoterms (Solid Vertical-skewed lines)
    for iso_t in range(-60, 60, 10):
        ax1.plot(get_skew_x(iso_t, p_ref), p_ref, color='black', alpha=0.1, lw=0.6)

    # B. Dry Adiabats (1°C per 100m)
    for t_start in range(-20, 100, 10):
        t_adiabat = t_start - (0.0098 * z_ref) 
        ax1.plot(get_skew_x(t_adiabat, p_ref), p_ref, color='orangered', alpha=0.15, lw=0.8)

    # C. Mixing Ratio (g/kg)
    for w in [2, 5, 10, 20]:
        e = (w * p_ref) / (622 + w)
        ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
        t_m = (243.5 * ln_e) / (17.67 - ln_e)
        ax1.plot(get_skew_x(t_m, p_ref), p_ref, color='green', alpha=0.15, ls='--', lw=0.8)

    # 4. PLOT SOUNDING (Correctly skewed data)
    ax1.plot(get_skew_x(t, p), p, color='red', linewidth=3, label='Temperature')
    ax1.plot(get_skew_x(td, p), p, color='green', linewidth=2, linestyle='--', label='Dewpoint')

    # 5. WIND PROFILE (km/h)
    ax2.plot(ws, p, color='red', linewidth=2)
    ax2.fill_betweenx(p, 0, ws, color='red', alpha=0.05)
    ax2.set_xlim(0, 100) # Max 100 km/h

    # 6. ALTITUDE & GRID SYNC
    km_all = np.arange(0, 9.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_all * 1000))**5.25588

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        # Remove all text/ticks from right side of main plot and left side of wind plot
        ax.tick_params(which='both', labelleft=False, labelright=False, left=False, right=False)
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1, ls='-')

    # Custom Altitude Labels (Placed between the two charts)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])}" if km_all[pos] % 1 == 0 else ""))
    ax1.tick_params(axis='y', labelleft=True, left=True)

    # 7. RANGE & FINAL LABELS
    # Shift X-limits to encompass the skewed profile and expand left buffer
    ax1.set_xlim(get_skew_x(-50, P_BOT), get_skew_x(40, P_BOT))
    ax1.set_ylabel("km", rotation=0, loc='top', labelpad=-20)
    ax1.set_xlabel("°C")
    ax2.set_xlabel("km/h")
    
    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Sounding Payerne | {ref_time} UTC", fontsize=16, y=0.94)
    ax1.legend(loc='upper left', frameon=True)

    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Tall, skewed, pressure-label-free emagram generated.")

if __name__ == "__main__":
    main()
