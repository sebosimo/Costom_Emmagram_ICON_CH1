import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import os, datetime, glob
import numpy as np

# --- Configuration ---
CACHE_DIR = "cache_data"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_plot(file_path):
    """Processes a single NetCDF file and saves a Skew-T plot."""
    try:
        ds = xr.open_dataset(file_path)
        
        # Metadata and Time Handling
        loc_name = ds.attrs.get("location", "Unknown")
        ref_dt = datetime.datetime.fromisoformat(ds.attrs["ref_time"])
        valid_dt = datetime.datetime.fromisoformat(ds.attrs["valid_time"])
        horizon_h = ds.attrs.get("horizon_h", 0)
        
        # Swiss Local Time (UTC+1 for Winter)
        swiss_dt = valid_dt + datetime.timedelta(hours=1) 

        # 1. Extract values
        p = ds["P"].values.squeeze() * units.Pa
        t = (ds["T"].values.squeeze() * units.K).to(units.degC)
        u_ms = ds["U"].values.squeeze() * units('m/s')
        v_ms = ds["V"].values.squeeze() * units('m/s')
        
        # Calculate Dewpoint from Specific Humidity (QV)
        qv = ds["QV"].values.squeeze() * units('kg/kg')
        td = mpcalc.dewpoint_from_specific_humidity(p, t, qv)

        z = mpcalc.pressure_to_height_std(p).to(units.km)
        wind_kmh = mpcalc.wind_speed(u_ms, v_ms).to('km/h').m

        # Sorting and Masking (to 7km for aviation)
        inds = z.argsort()
        z_plot, t_plot, td_plot = z[inds].m, t[inds].m, td[inds].m
        u_plot, v_plot, wind_plot = u_ms[inds].to('km/h').m, v_ms[inds].to('km/h').m, wind_kmh[inds]
        
        mask = z_plot <= 7.0
        z_plot, t_plot, td_plot = z_plot[mask], t_plot[mask], td_plot[mask]
        u_plot, v_plot, wind_plot = u_plot[mask], v_plot[mask], wind_plot[mask]

        # Skew Configuration
        SKEW_FACTOR = 5 
        def skew_x(temp, height): return temp + (height * SKEW_FACTOR)
        skew_t, skew_td = skew_x(t_plot, z_plot), skew_x(td_plot, z_plot)

        # 2. Figure Setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10), sharey=True, 
                                       gridspec_kw={'width_ratios': [3, 1], 'wspace': 0})
        
        # Dynamic X-Axis
        padding = 8
        ax1.set_xlim(min(np.min(skew_t), np.min(skew_td)) - padding, max(np.max(skew_t), np.max(skew_td)) + padding)
        ax1.set_ylim(0, 7.0)

        # Draw helper grid (Isotherms)
        for temp_base in range(-150, 151, 5):
            ax1.plot([skew_x(temp_base, 0), skew_x(temp_base, 7)], [0, 7], color='blue', alpha=0.06, zorder=1)

        # Plot Temp (colored by lapse rate) and Dewpoint
        dt, dz = np.diff(t_plot), np.diff(z_plot)
        lapse_rate = - (dt / dz) 
        points = np.array([skew_t, z_plot]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='RdYlGn', norm=Normalize(vmin=-3, vmax=10), linewidth=4, zorder=5)
        lc.set_array(lapse_rate)
        ax1.add_collection(lc)
        ax1.plot(skew_td, z_plot, color='blue', linewidth=2, zorder=5, alpha=0.8)

        # Panel 2: Wind
        ax2.plot(wind_plot, z_plot, color='blue', alpha=0.3)
        ax2.set_xlim(0, 100) 
        ax2.set_xlabel("Wind (km/h)")
        step = max(1, len(z_plot) // 15) 
        ax2.barbs(np.ones_like(z_plot[::step]) * 85, z_plot[::step], u_plot[::step], v_plot[::step], length=6)

        # Accurate Title Section
        title_line1 = f"{loc_name}  |  ICON-CH1 Run: {ref_dt.strftime('%d.%m. %H:%M')} UTC"
        title_line2 = (f"FORECAST FOR: {valid_dt.strftime('%H:%M')} UTC "
                       f"({swiss_dt.strftime('%H:%M')} Local Time)  |  Lead: +{horizon_h}h")
        
        fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=14, fontweight='bold', y=0.97)
        ax1.set_ylabel("Altitude (km MSL)")
        ax1.set_xlabel("Temperature (°C)")

        # Save and Close
        out_name = f"{loc_name}_H{horizon_h:02d}.png".replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=120, bbox_inches='tight')
        plt.close(fig) 
        return True
    except Exception as e:
        print(f"Error plotting {file_path}: {e}")
        return False

def main():
    files = glob.glob(os.path.join(CACHE_DIR, "*.nc"))
    if not files:
        print("No data found to plot.")
        return

    print(f"Generating plots for {len(files)} files...")
    for f in sorted(files):
        generate_plot(f)
    print(f"Done. Plots are in the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()
