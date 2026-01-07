import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob
import numpy as np

CACHE_DIR = "cache_data"

def format_pressure_as_km(x, pos):
    """Converts Pressure (hPa) to height in km."""
    if x <= 0: return ""
    height = mpcalc.pressure_to_height_std(x * units.hPa).to('km')
    return f"{height.m:.1f}"

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
    u, v = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    # 2. Dewpoint Calculation
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # 3. Sort Surface -> Space
    inds = p.argsort()[::-1]
    p_hpa, t, td, u, v = p[inds].to(units.hPa), t[inds], td[inds], u[inds], v[inds]

    # 4. Visualization Setup
    fig = plt.figure(figsize=(12, 12))
    
    # We use rotation=0 to make it a vertical chart (easier for lapse rates)
    # We use logarithmic pressure (mimics linear height perfectly)
    skew = SkewT(fig, rotation=30) # A slight skew (30 instead of 45) helps readability
    
    # 5. Set Limits: Focus on lower atmosphere (Paragliding levels)
    skew.ax.set_ylim(1050, 400) # Surface to ~7km
    skew.ax.set_xlim(-20, 35)   # Relevant temp range for flight

    # 6. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    skew.plot_barbs(p_hpa[::2], u[::2], v[::2], xloc=1.05) # Barbs on the right side
    
    # 7. Add Paragliding Indicators
    # Mark Freezing Level
    zero_crossings = np.where(np.diff(np.sign(t.m)))[0]
    if zero_crossings.size > 0:
        p_zero = p_hpa[zero_crossings[0]]
        skew.ax.axhline(p_zero, color='blue', linestyle='--', alpha=0.3)
        skew.ax.text(-18, p_zero, " 0°C (Freezing Level)", color='blue', alpha=0.7)

    # 8. Background Lines (Adiabats)
    skew.plot_dry_adiabats(alpha=0.2, color='orangered', linewidth=1)
    skew.plot_moist_adiabats(alpha=0.2, color='blue', linewidth=1)
    skew.plot_mixing_lines(alpha=0.2, color='green', linestyle=':')
    
    # 9. Dual Axis Labels
    skew.ax.set_ylabel("Altitude (km) [Std. Atmosphere]")
    skew.ax.set_xlabel("Temperature (°C)")
    
    pressure_ticks = [1000, 900, 850, 800, 750, 700, 600, 500, 400]
    skew.ax.set_yticks(pressure_ticks)
    skew.ax.yaxis.set_major_formatter(FuncFormatter(format_pressure_as_km))
    
    # Right-side pressure axis
    ax_hpa = skew.ax.twinx()
    ax_hpa.set_yscale('log') # Must match SkewT internal scale
    ax_hpa.set_ylim(1050, 400)
    ax_hpa.set_yticks(pressure_ticks)
    ax_hpa.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    ax_hpa.set_ylabel("Pressure (hPa)")

    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=16, pad=20)
    skew.ax.legend(loc='upper left')
    
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: Readable Skew-T generated.")

if __name__ == "__main__":
    main()
