import os
# ... (all your other imports)

# Configuration for caching
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def main():
    print("Fetching ICON-CH1 data...")
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]
    
    success, profile_data, ref_time_final = False, {}, None

    for ref_time in times_to_try:
        time_tag = ref_time.strftime('%Y%m%d_%H%M')
        cache_path = os.path.join(CACHE_DIR, f"profile_{time_tag}.nc")

        # Check if we already have this data locally
        if os.path.exists(cache_path):
            print(f">>> Loading {time_tag} from CACHE...")
            ds_cache = xr.open_dataset(cache_path)
            for var in CORE_VARS + ["HUM"]:
                profile_data[var] = ds_cache[var]
            profile_data["HUM_TYPE"] = ds_cache.attrs["HUM_TYPE"]
            success, ref_time_final = True, ref_time
            break

        print(f"--- Attempting Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            # ... (Your existing download logic) ...
            # [Keep your loop for CORE_VARS and Humidity here]
            
            # If download was successful, save it to cache
            if success:
                # Combine variables into a single dataset for storage
                ds_to_save = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
                ds_to_save.attrs["HUM_TYPE"] = profile_data["HUM_TYPE"]
                ds_to_save.to_netcdf(cache_path)
                print(f">>> Saved {time_tag} to cache.")
                break

        except Exception as e:
            print(f"Run incomplete: {e}")

    if not success:
        print("Error: No complete model runs found.")
        return

    # ... (Rest of your plotting code stays exactly the same) ...
