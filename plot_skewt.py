import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob

CACHE_DIR = "cache_data"

def format_pressure_as_km(x, pos):
    """Converts Pressure (hPa) to height in km for the linear axis."""
    if x <= 0: return ""
    height = mpcalc.pressure_to_height_std(x * units.hPa).to('km')
    return f"{height.m:.1f}"

def main():
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found. Run fetch_data.py first.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Plotting paragliding-focused view from: {latest_file}")
    ds = xr.open_dataset(latest_file)
    
    # 1. Extract values and assign units
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
    # We use a standard SkewT object but will force the axis to be linear
    skew = SkewT(fig, rotation=45)
    
    # Force the Y-axis to be LINEAR
    skew.ax.set_yscale('linear')
    
    # Set Limits: Paraglider focus (Surface to 300 hPa)
    skew.ax.set_ylim(1050, 300) 
    skew.ax.set_xlim(-30, 40) # Focused temperature range

    # 5. Plotting Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Plot barbs - decimated for the lower atmosphere focus
    skew.plot_barbs(p_hpa[::2], u[::2], v[::2], x_offset=1.1)
    
    # 6. Linear Supporting Lines
    # SkewT plot helpers automatically adapt to the axis scale
    skew.plot_dry_adiabats(alpha=0.2, color='orangered', linewidth=1)
    skew.plot_moist_adiabats(alpha=0.2, color='blue', linewidth=1)
    skew.plot_mixing_lines(alpha=0.2, color='green', linestyle='--')
    
    # 7. Labels and Formatting
    skew.ax.set_ylabel("Altitude (km) [Std. Atmosphere]")
    skew.ax.set_xlabel("Temperature (°C)")
    
    # Set specific Y-ticks for clarity in the lower levels
    pressure_ticks = [1000, 900, 800, 700, 600, 500, 400, 300]
    skew.ax.set_yticks(pressure_ticks)
    skew.ax.yaxis.set_major_formatter(FuncFormatter(format_pressure_as_km))
    
    # Add a second Y-axis on the right to show raw hPa values
    ax_hpa = skew.ax.twinx()
    ax_hpa.set_yscale('linear')
    ax_hpa.set_ylim(1050, 300)
    ax_hpa.set_yticks(pressure_ticks)
    ax_hpa.set_ylabel("Pressure (hPa)")

    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Payerne Paragliding Sounding (Linear) | {ref_time_str} UTC", fontsize=16, pad=20)
    skew.ax.legend(loc='upper left', frameon=True)
    
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: Linear Skew-T generated for the lower atmosphere.")

if __name__ == "__main__":
    main()
