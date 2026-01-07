import os, sys, datetime, xarray as xr
from meteodatalab import ogd_api
import numpy as np

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_data(ds):
    """Deep cleans attributes to ensure NetCDF compatibility."""
    if ds is None: return None
    # Convert to dataset if it's a DataArray
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()
    
    # Remove all complex metadata objects
    ds.attrs = {}
    for var in list(ds.data_vars) + list(ds.coords):
        ds[var].attrs = {k: v for k, v in ds[var].attrs.items() 
                        if isinstance(v, (str, int, float, np.ndarray))}
    return ds

def get_nearest_profile(ds, lat_target, lon_target):
    """Extracts a vertical profile safely."""
    if ds is None: return None
    
    # Handle cases where API returns a list
    if isinstance(ds, list):
        if not ds: return None
        ds = ds[0]
        
    try:
        data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
        lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
        lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
        
        # Calculate distance
        dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
        flat_idx = int(dist.argmin())
        
        # Extract and compute
        horiz_dims = data.coords[lat_coord].dims
        if len(horiz_dims) == 1:
            profile = data.isel({horiz_dims[0]: flat_idx})
        else:
            profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
            
        return profile.squeeze().compute()
    except Exception as e:
        print(f"      ! Extraction failed: {e}")
        return None

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
            print(f">>> {time_tag} already in cache. Skipping.")
            return

        print(f"--- Attempting Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            profile_data = {}
            # Fetch Core Variables
            for var in CORE_VARS:
                print(f"  > Fetching {var}...")
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var, 
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None: raise ValueError(f"Could not get {var}")
                profile_data[var] = res
            
            # Fetch Humidity
            print(f"  > Fetching Humidity...")
            for hum in ["RELHUM", "QV"]:
                req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum, 
                                       reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                if res_h is not None:
                    profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum
                    break
            
            if "HUM" not in profile_data: raise ValueError("No humidity data")

            # Finalize and Save
            ds = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
            ds = clean_data(ds)
            ds.attrs["HUM_TYPE"] = profile_data["HUM_TYPE"]
            ds.attrs["ref_time"] = ref_time.isoformat()
            
            ds.to_netcdf(cache_path)
            print(f">>> SUCCESS: Data for {time_tag} saved to cache.")
            return # We are done!
            
        except Exception as e:
            print(f"  ! Run {ref_time.strftime('%H:%M')} incomplete: {e}")
            continue

    print("ERROR: No complete model runs found.")
    sys.exit(1)

if __name__ == "__main__":
    main()
