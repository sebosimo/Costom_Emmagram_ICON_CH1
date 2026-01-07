import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob

# Configuration
CACHE_DIR = "cache_data"

def main():
    # 1. Find the newest data file
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("Error: No cached data found. Run fetch_data.py first.")
        return
    
    latest_file = max(files, key=os.path.getctime)
    print(f"Generating plot from: {latest_file}")
    
    # 2. Load and extract data
    ds = xr.open_dataset(latest_file)
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')

    p = ds["P"].values * units.Pa
    t = (ds["T"].values * units.K)
    u, v = ds["U"].values * units('m/s'), ds["V"].values * units('m/s')
    
    # Dewpoint calculation based on humidity type
    if ds.attrs["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, ds["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # Sort descending for Skew-T (Surface -> Space)
    inds = p.argsort()[::-1]
    p_hpa, t_degc, td_degc = p[inds].to(units.hPa), t[inds].to(units.degC), td[inds].to(units.degC)
    u, v = u[inds], v[inds]

    # 3. Plotting Logic
    fig = plt.figure(figsize=(10, 12))
    skew = SkewT(fig, rotation=45)
    
    skew.plot(p_hpa, t_degc, 'r', linewidth=2.5, label='Temperature')
    skew.plot(p_hpa, td_degc, 'g', linewidth=2.5, label='Dewpoint')
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3])
    
    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    plt.title(f"ICON-CH1 Sounding (Payerne) | {ref_time_str} UTC", fontsize=14)
    skew.ax.legend(loc='upper left')
    
    # 4. Save
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("Success: latest_skewt.png generated.")

if __name__ == "__main__":
    main()
