import os, sys, datetime, xarray as xr
from meteodatalab import ogd_api
import metpy.calc as mpcalc
from metpy.units import units

LAT_TARGET, LON_TARGET = 46.81, 6.94
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_for_netcdf(ds):
    """Removes non-serializable earthkit metadata objects."""
    ds.attrs = {k: v for k, v in ds.attrs.items() if isinstance(v, (str, int, float, list, tuple))}
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].attrs = {}
    return ds

def get_nearest_profile(ds, lat, lon):
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_c = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_c = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_c].dims
    dist = (data[lat_c] - lat)**2 + (data[lon_c] - lon)**2
    idx = dist.argmin().values
    profile = data.isel({horiz_dims[0]: idx}) if len(horiz_dims) == 1 else data.stack(gp=horiz_dims).isel(gp=idx)
    return profile.squeeze().compute()

def main():
    force = os.getenv("FORCE_REFRESH") == "true"
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]

    for ref_time in times_to_try:
        time_tag = ref_time.strftime('%Y%m%d_%H%M')
        cache_path = os.path.join(CACHE_DIR, f"profile_{time_tag}.nc")

        if os.path.exists(cache_path) and not force:
            print(f">>> {time_tag} already cached. Skipping download.")
            return # Exit early, we have what we need

        print(f"--- Downloading Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            profile_data = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var, reference_datetime=ref_time, horizon="P0DT0H")
                profile_data[var] = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
            
            for hum in ["RELHUM", "QV"]:
                req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum, reference_datetime=ref_time, horizon="P0DT0H")
                res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                if res_h is not None:
                    profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum
                    break
            
            ds = clean_for_netcdf(xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]}))
            ds.attrs["HUM_TYPE"] = profile_data["HUM_TYPE"]
            ds.attrs["ref_time"] = ref_time.isoformat()
            ds.to_netcdf(cache_path)
            print(f">>> SUCCESS: Saved {cache_path}")
            return
        except Exception as e:
            print(f"Run {ref_time.strftime('%H:%M')} failed: {e}")

    sys.exit(1)

if __name__ == "__main__":
    main()
