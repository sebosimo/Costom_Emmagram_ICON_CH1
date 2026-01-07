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
    p_raw = ds["P"].values / 100.0  # Convert Pa to hPa
    t_raw = ds["T"].values - 273.15 # Convert K to C
    u = ds["U"].values
    v = ds["V"].values
    ws_kmh = np.sqrt(u**2 + v**2) * 3.6 # Wind speed in km/h

    # Sort data from Surface Upwards
    idx = p_raw.argsort()[::-1]
    p, t, ws = p_raw[idx], t_raw[idx], ws_kmh[idx]

    # Handle Dewpoint
    hum = ds["HUM"].values[idx]
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        # Simple approximation for Dewpoint from RH
        td = t - ((100 - hum) / 5)
    else:
        td = t # Placeholder if specific humidity is complex

    # --- SKEW CONFIGURATION ---
    # Adjust SKEW_X to change the "lean" (higher = more lean to the right)
    SKEW_X = 45 
    P_BOT, P_TOP = 1020, 400

    def skew_x(temp, press):
        return temp + SKEW_X * np.log10(1020 / press)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(12, 14))
    # Panel 1: Emagram (Left), Panel 2: Wind (Right)
    ax1 = fig.add_axes([0.1, 0.1, 0.6, 0.8])
    ax2 = fig.add_axes([0.7, 0.1, 0.15, 0.8])

    # 3. BACKGROUND HELPER LINES
    # A. Dry Adiabats (1°C per 100m)
    # T_z = T_surface - (lapse * z) -> approximated via pressure
    for t_start in range(-20, 70, 10):
        # Create a line following 1C/100m lapse rate
        z_ref = np.linspace(0, 8000, 50)
        p_ref = 1013.25 * (1 - 2.25577e-5 * z_ref)**5.25588
        t_ref = t_start - (0.0098 * z_ref) # ~1C per 100m
        ax1.plot(skew_x(t_ref, p_ref), p_ref, color='orangered', alpha=0.15, linewidth=0.8)

    # B. Mixing Ratio (Using your provided logic)
    w_values = [2, 5, 10, 20]
    for w in w_values:
        z_m = np.linspace(0, 8000, 50)
        p_m = 1013.25 * (1 - 2.25577e-5 * z_m)**5.25588
        e = (w * p_m) / (622 + w)
        ln_e = np.log(np.clip(e / 6.112, 1e-5, None))
        t_m = (243.5 * ln_e) / (17.67 - ln_e)
        ax1.plot(skew_x(t_m, p_m), p_m, color='green', alpha=0.15, linestyle='--', linewidth=0.8)

    # 4. PLOT ACTUAL SOUNDING
    ax1.plot(skew_x(t, p), p, color='red', linewidth=3, label='Temp')
    ax1.plot(skew_x(td, p), p, color='green', linewidth=2, linestyle='--', label='Dewp')

    # 5. PANEL 2: WIND SPEED
    ax2.plot(ws, p, color='blue', linewidth=2)
    ax2.fill_betweenx(p, 0, ws, color='blue', alpha=0.1)
    ax2.set_xlim(0, 80)
    ax2.set_xlabel("km/h", fontsize=12)

    # 6. ALTITUDE & COORDINATE SYNC
    km_levels = np.arange(0, 8.5, 0.5)
    p_levels = 1013.25 * (1 - 2.25577e-5 * (km_levels * 1000))**5.25588

    for ax in [ax1, ax2]:
        ax.set_yscale('log')
        ax.set_ylim(P_BOT, P_TOP)
        ax.yaxis.set_major_locator(FixedLocator(p_levels))
        ax.grid(True, which='major', axis='y', color='black', alpha=0.1)

    # Label only ax1 with km, hide ax2 y-ticks
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_levels[pos])} km" if km_levels[pos] % 1 == 0 else ""))
    ax2.set_yticklabels([])

    # 7. FINAL TOUCHES
    ax1.set_xlim(skew_x(-40, 1020), skew_x(40, 1020))
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    
    ref_time = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%d.%m.%Y %H:%M')
    plt.suptitle(f"Paraglider Emagram Payerne | {ref_time} UTC", fontsize=16, y=0.95)
    ax1.legend(loc='upper left')

    plt.savefig("latest_skewt.png", dpi=150, bbox_inches='tight')
    print("Custom Emagram generated successfully.")

if __name__ == "__main__":
    main()
