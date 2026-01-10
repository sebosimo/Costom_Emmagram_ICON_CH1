import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
# We use QV (Specific Humidity) because it's guaranteed to be on the same 
# 80 model levels as T, U, V, and P.
CORE_VARS = ["T", "U", "V", "P", "QV"] 
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    indices = {}
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    for name, coords in locations.items():
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        idx = int(np.argmin(dist))
        indices[name] = idx
    return indices

def main():
    if not os.path.exists("locations.json"): return
    with open("locations.json", "r") as f: locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    cached_indices = None

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        locs_to_do = [n for n in locations.keys() 
                      if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_{time_tag}_H{h_int:02d}.nc"))]
        if not locs_to_do: continue

        print(f"\nHorizon +{h_int:02d}h: Fetching domain...")
        domain_data = {}

        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_data[var] = ogd_api.get_from_ogd(req)

            if cached_indices is None:
                cached_indices = get_location_indices(domain_data["T"], locations)

            for name in locs_to_do:
                idx = cached_indices[name]
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")
                
                loc_vars = {}
                for var in CORE_VARS:
                    ds_field = domain_data[var]
                    # Identify the spatial dimension (ncells, cell, or values)
                    spatial_dim = ds_field.dims[-1] 
                    for d in ['ncells', 'cell', 'values']:
                        if d in ds_field.dims:
                            spatial_dim = d
                            break
                    
                    # Extract 1D profile and drop all complex coordinates that break merging
                    profile = ds_field.isel({spatial_dim: idx}).squeeze().compute()
                    # We drop coordinates to force alignment by index
                    loc_vars[var] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

                # Merge by index (guarantees Level 1 T matches Level 1 P and Level 1 QV)
                ds_final = xr.Dataset(loc_vars)
                ds_final.attrs = {
                    "location": name, 
                    "HUM_TYPE": "QV", 
                    "ref_time": ref_time.isoformat(), 
                    "horizon_h": h_int,
                    "valid_time": valid_time.isoformat()
                }
                
                # Strip all remaining metadata that causes issues
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h failed: {e}")

if __name__ == "__main__":
    main()
