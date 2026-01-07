import os, sys, datetime, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94  # Payerne
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_extract_profile(api_res, lat_target, lon_target):
    """Safely handles earthkit responses and extracts a vertical profile."""
    # Strict check to avoid "Ambiguous Truth Value" error
    if api_res is None:
        return None
    if isinstance(api_res, list) and len(api_res) == 0:
        return None

    try:
        # 1. Get the first item if it's a list/fieldset
        ds_raw = api_res[0] if hasattr(api_res, "__getitem__") else api_res
        
        # 2. Convert to xarray
        ds = ds_raw.to_xarray() if hasattr(ds_raw, "to_xarray") else ds_raw
        if isinstance(ds, list): ds = ds[0]

        # 3. Coordinate standardization
        lat_n = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_n = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        # 4. Find the horizontal index
        dist = (ds[lat_n] - lat_target)**2 + (ds[lon_n] - lon_target)**2
        idx = int(dist.argmin())
        
        # 5. Extract column (isel requires the name of the horizontal dimension)
        horiz_dim = ds[lat_n].dims[0]
        profile = ds.isel({horiz_dim: idx}).squeeze().compute()
        
        # 6. Aggressive cleanup for NetCDF serialization
        profile.attrs = {}
        for c in profile.coords:
            profile[c].attrs = {}
            
        return profile
    except Exception as e:
        print(f"      ! Extraction error: {e}")
        return None

def main():
    force = os.getenv("FORCE_REFRESH") == "true"
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Try the last 4 runs (0h, -3h, -6h, -9h)
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]

    for ref_time in times_to_try:
        time_tag = ref_time.strftime('%Y%m%d_%H%M')
        cache_path = os.path.join(CACHE_DIR, f"profile_{time_tag}.nc")

        if os.path.exists(cache_path) and not force:
            print(f">>> Run {time_tag} found in cache. Skipping.")
            return

        print(f"--- Checking Availability: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            # Check T first. If T isn't there, don't even try the others.
            req_t = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T", 
                                   reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
            res_t = safe_extract_profile(ogd_api.get_from_ogd(req_t), LAT_TARGET, LON_TARGET)
            
            # Use isinstance check to avoid ambiguous truth value error
            if not isinstance(res_t, xr.DataArray):
                print(f"  ! Run {time_tag} is not yet complete on server. Trying older...")
                continue

            profile_data = {"T": res_t}

            # Fetch remaining Core Vars
            for var in ["U", "V", "P"]:
                print(f"  > Fetching {var}...")
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var, 
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = safe_extract_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if not isinstance(res, xr.DataArray): raise ValueError(f"{var} missing")
                profile_data[var] = res
            
            # Fetch Humidity
            hum_found = False
            for hum_var in ["RELHUM", "QV"]:
                print(f"  > Trying {hum_var}...")
                req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var, 
                                       reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res_h = safe_extract_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                if isinstance(res_h, xr.DataArray):
                    profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                    hum_found = True
                    break
            
            if not hum_found: raise ValueError("Humidity missing")

            # Save to Cache
            ds_final = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
            ds_final.attrs = {"HUM_TYPE": profile_data["HUM_TYPE"], "ref_time": ref_time.isoformat()}
            
            ds_final.to_netcdf(cache_path)
            print(f">>> SUCCESS: Cached {cache_path}")
            return 

        except Exception as e:
            print(f"  ! Run {ref_time.strftime('%H:%M')} failed: {e}")
            continue

    sys.exit(1)

if __name__ == "__main__":
    main()
