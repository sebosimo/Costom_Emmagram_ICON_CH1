import os
import xarray as xr
# ... (rest of your imports)

DATA_DIR = "cache_data"
os.makedirs(DATA_DIR, exist_ok=True)

def main():
    print("Fetching ICON-CH1 data...")
    # ... (your existing time calculation logic)

    success, profile_data, ref_time_final = False, {}, None
    for ref_time in times_to_try:
        time_str = ref_time.strftime('%Y%m%d%H%M')
        cache_file = os.path.join(DATA_DIR, f"profile_{time_str}.nc")
        
        # --- CACHING LOGIC ---
        if os.path.exists(cache_file):
            print(f"Loading cached data for {time_str}...")
            cached_ds = xr.open_dataset(cache_file)
            # Map variables back to your dictionary
            for var in CORE_VARS + ["HUM"]:
                profile_data[var] = cached_ds[var]
            profile_data["HUM_TYPE"] = cached_ds.attrs["HUM_TYPE"]
            success, ref_time_final = True, ref_time
            break

        print(f"--- Attempting Download: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            # (Your existing download logic)
            # ... loop through variables ...
            
            # --- SAVE TO CACHE IF SUCCESSFUL ---
            if success:
                # Combine into one dataset for easy storage
                ds_to_save = xr.Dataset(profile_data)
                ds_to_save.attrs["HUM_TYPE"] = profile_data["HUM_TYPE"]
                ds_to_save.to_netcdf(cache_file)
                print(f"Saved to cache: {cache_file}")
                break 
        except Exception as e: 
            print(f"Run incomplete: {e}")
