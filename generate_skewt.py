import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
from meteodatalab import ogd_api
import xarray as xr
import numpy as np
import datetime
import os
import sys

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_for_netcdf(ds):
    """Removes non-serializable earthkit metadata objects from xarray Dataset."""
    # Strip attributes from the dataset itself
    ds.attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float, list, tuple))}
    # Strip attributes from every variable and coordinate
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].attrs = {}
    return ds

def get_nearest_profile(ds, lat_target, lon_target):
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    if len(horiz_dims) == 1:
        profile = data.isel({horiz_dims[0]: flat_idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
    return profile.squeeze().compute()

def format_pressure_as_km(x, pos):
    if x <= 0: return ""
    height = mpcalc.pressure_to_height_std(x * units.hPa).to('km')
    return f"{height.m:.1f}"

def main():
    print(f"--- Process Started at: {datetime.datetime.now(datetime.timezone.utc)} UTC ---")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success, profile_data, ref_time_final = False, {}, None

    for ref_time in times_to_try:
        time_tag = ref_time.strftime('%Y%m%d_%H%M')
        cache_path = os.path.join(CACHE_DIR, f"profile_{time_tag}.nc")

        if os.path.exists(cache_path):
            print(f">>> DEBUG: Loading {time_tag} from CACHE...")
            try:
                ds_cache = xr.open_dataset(cache_path)
                for var in CORE_VARS + ["HUM"]:
                    profile_data[var] = ds_cache[var].load()
                profile_data["HUM_TYPE"] = ds_cache.attrs["HUM_TYPE"]
                success, ref_time_final = True, ref_time
                break 
            except Exception as e:
                print(f"WARNING: Cache load failed: {e}")

        print(f"--- DEBUG: Attempting Download for Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None: raise ValueError(f"Empty {var}")
                profile_data[var] = res
            
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                           reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None:
                        profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" not in profile_data: raise ValueError("No Humidity found")
            
            # --- CRITICAL FIX: DATA IS GOOD, MARK SUCCESS FIRST ---
            success, ref_time_final = True, ref_time
            
            # Now try to save cache, but don't fail the whole script if caching fails
            try:
                ds_to_save = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
                ds_to_save = clean_for_netcdf(ds_to_save) # Strip metadata objects
                ds_to_save.attrs["HUM_TYPE"] = profile_data["HUM_TYPE"]
                ds_to_save.to_netcdf(cache_path)
                print(f">>> SUCCESS: Cached {time_tag}")
            except Exception as cache_e:
                print(f"WARNING: Could not save cache: {cache_e}")
            
            break # We have the latest available data, stop looking for older runs

        except Exception as e: 
            print(f"FAILURE: Run {ref_time.strftime('%H:%M')} unavailable: {e}")

    if not success:
        print("CRITICAL ERROR: No data found.")
        sys.exit(1)

    # --- Plotting Code ---
    print(f"DEBUG: Plotting for {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC")
    p = profile_data["P"].values * units.Pa
    t = (profile_data["T"].values * units.K).to(units.degC)
    u = profile_data["U"].values * units('m/s')
    v = profile_data["V"].values * units('m/s')
    
    if profile_data["HUM_TYPE"] == "RELHUM":
        td = mpcalc.dewpoint_from_relative_humidity(t, profile_data["HUM"].values / 100.0)
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, profile_data["HUM"].values * units('kg/kg'))

    inds = p.argsort()[::-1] 
    p, t, td, u, v = p[inds], t[inds], td[inds], u[inds], v[inds]
    p_hpa = p.to(units.hPa)
    wind_speed = mpcalc.wind_speed(u, v).to(units('km/h'))
    parcel_prof = mpcalc.parcel_profile(p_hpa, t[0], td[0]).to('degC')

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    skew = SkewT(fig, rotation=45, subplot=gs[0])

    skew.plot(p_hpa, t, 'r', linewidth=2.5, label='Temperature')
    skew.plot(p_hpa, td, 'g', linewidth=2.5, label='Dewpoint')
    skew.plot(p_hpa, parcel_prof, 'k', linestyle='--', linewidth=1.5, label='Surface Parcel')
    skew.shade_cape(p_hpa, t, parcel_prof)
    skew.shade_cin(p_hpa, t, parcel_prof, td)
    skew.plot_barbs(p_hpa[::5], u[::5], v[::5])

    skew.plot_dry_adiabats(alpha=0.1, color='red')
    skew.plot_moist_adiabats(alpha=0.1, color='blue')
    skew.plot_mixing_lines(alpha=0.1, color='green')
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlim(-40, 40)
    
    ax_wind = fig.add_subplot(gs[1], sharey=skew.ax)
    ax_wind.plot(wind_speed, p_hpa, color='purple', linewidth=2)
    ax_wind.set_xlabel("Wind Speed [km/h]")
    ax_wind.grid(True, which='both', linestyle='--', alpha=0.5)
    ax_wind.set_yscale('log')
    ax_wind.set_ylim(1050, 100)
    plt.setp(ax_wind.get_yticklabels(), visible=False)
    
    skew.ax.set_ylabel("Altitude (km) [Std. Atm]")
    pressure_levels = [1000, 850, 700, 500, 400, 300, 200, 100]
    skew.ax.set_yticks(pressure_levels)
    skew.ax.yaxis.set_major_formatter(FuncFormatter(format_pressure_as_km))
    
    plt.suptitle(f"ICON-CH1 Sounding (Payerne) | {ref_time_final.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=16, y=0.92)
    skew.ax.legend(loc='upper left')
    plt.savefig("latest_skewt.png", bbox_inches='tight', dpi=150)
    print("COMPLETE: latest_skewt.png generated.")

if __name__ == "__main__":
    main()
