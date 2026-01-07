import os, sys, datetime, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_extract_profile(api_res, lat_target, lon_target):
    """Handles the list-style response from the OGD API safely."""
    if not api_res:
        return None
    
    # If the API returns a list, take the first valid model field
    ds_raw = api_res[0] if isinstance(api_res, list) else api_res
    
    # Convert to xarray if it isn't already
    ds = ds_raw.to_xarray() if hasattr(ds_raw, "to_xarray") else ds_raw
    
    # Standardize coordinate names
    lat_n = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    # Find nearest point
    dist = (ds[lat_n] - lat_target)**2 + (ds[lon_n] - lon_target)**2
    idx = int(dist.argmin())
    
    # Extract vertical column and strip all non-serializable metadata
    horiz_dim = ds[lat_n].dims[0]
    profile = ds.isel({horiz_dim: idx}).squeeze().compute()
    
    # Clear all attributes that break NetCDF saving
    profile.attrs = {}
    for coord in profile.coords:
        profile[coord].attrs = {}
        
    return profile

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
            print(f">>> Run {time_tag} already cached. Skipping.")
            return

        print(f"--- Attempting Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            profile_data = {}
            
            # Fetch Core Variables
            for var in CORE_VARS:
                print(f"  > Fetching {var}...")
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var, 
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = safe_extract_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None: raise ValueError(f"Missing variable: {var}")
                profile_data[var] = res
            
            # Fetch Humidity (fallback logic)
            hum_found = False
            for hum_var in ["RELHUM", "QV"]:
                print(f"  > Trying Humidity: {hum_var}...")
                req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var, 
                                       reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res_h = safe_extract_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                if res_h is not None:
                    profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                    hum_found = True
                    break
            
            if not hum_found: raise ValueError("No humidity variables available.")

            # Create final Dataset
            ds_final = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
            ds_final.attrs = {
                "HUM_TYPE": profile_data["HUM_TYPE"],
                "ref_time": ref_time.isoformat()
            }
            
            # Final Save
            ds_final.to_netcdf(cache_path)
            print(f">>> SUCCESS: Saved {cache_path}")
            return # Success! Stop the loop.

        except Exception as e:
            print(f"  ! Run {ref_time.strftime('%H:%M')} failed during processing: {e}")
            continue

    print("CRITICAL: All model run attempts failed.")
    sys.exit(1)

if __name__ == "__main__":
    main()
