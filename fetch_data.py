import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
# We use T, U, V, P and QV (Specific Humidity) for perfect vertical alignment
CORE_VARS = ["T", "U", "V", "P", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    """Converts integer hours to ISO8601 duration string."""
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    """Calculates nearest grid indices for all locations."""
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    indices = {}
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    
    for name, coords in locations.items():
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        indices[name] = idx
    return indices

def main():
    if not os.path.exists("locations.json"):
        print("Error: locations.json not found.")
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    # 1. Determine Model Run
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    cached_indices = None

    # 2. Main Loop: Iterate by Horizon
    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        # Filter only locations missing this specific time step
        locs_to_do = [n for n in locations.keys() 
                      if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_{time_tag}_H{h_int:02d}.nc"))]
        
        if not locs_to_do:
            continue

        print(f"\nHorizon +{h_int:02d}h: Fetching domain fields...")
        domain_fields = {}

        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields["T"], locations)

            for name in locs_to_do:
                idx = cached_indices[name]
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")

                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    # Robust dimension detection (ncells vs cell)
                    spatial_dim = None
                    for d in ['ncells', 'cell', 'values', 'index']:
                        if d in ds_field.dims:
                            spatial_dim = d
                            break
                    
                    if spatial_dim:
                        subset = ds_field.isel({spatial_dim: idx[0]})
                    else:
                        subset = ds_field.isel(y=idx[0], x=idx[1])
                    
                    # Compute and squeeze (removes 'number' ensemble dim)
                    # We drop extra coords to ensure a clean merge
                    res = subset.squeeze().compute()
                    loc_data[var_name] = res.drop_vars([c for c in res.coords if c not in res.dims])

                # Merge into dataset
                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {
                    "location": name,
                    "HUM_TYPE": "QV", 
                    "ref_time": ref_time.isoformat(),
                    "horizon_h": h_int,
                    "valid_time": valid_time.isoformat()
                }

                # Strip metadata that breaks NetCDF saving
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h failed: {e}")

if __name__ == "__main__":
    main()
