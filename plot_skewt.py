import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from metpy.plots import SkewT
from matplotlib.ticker import FixedLocator, FuncFormatter
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
@@ -29,69 +28,71 @@ def main():
    else:
        td = mpcalc.dewpoint_from_specific_humidity(p, t, ds["HUM"].values * units('kg/kg'))

    # Calculate wind speed in km/h
    wind_speed = mpcalc.wind_speed(u, v).to('km/h')
    # Sort Surface -> Space
    inds = p.argsort()[::-1]
    p_hpa, t, td, u, v = p[inds].to(units.hPa), t[inds], td[inds], u[inds], v[inds]
    p_hpa, t, td, wind_speed_val = p[inds].to(units.hPa), t[inds], td[inds], wind_speed[inds]

    # 3. Visualization Setup
    # Force a very tall figure
    fig = plt.figure(figsize=(8, 14))
    
    # Set manual margins to force the tall aspect ratio
    # left, bottom, right, top (0 to 1)
    plt.subplots_adjust(left=0.15, bottom=0.08, right=0.85, top=0.92)
    # 3. Figure Setup (Wide canvas to fit two panels)
    fig = plt.figure(figsize=(12, 14))

    skew = SkewT(fig, rotation=30)
    skew.ax.set_aspect('auto') # Fill the adjusted subplot area
    
    # --- RANGE ADJUSTMENTS ---
    # Shifted Right: -30 to +50 (Summer range)
    skew.ax.set_xlim(-30, 50)
    # Surface (1020) to ~7km (400)
    skew.ax.set_ylim(1020, 400) 
    # Define two panels: [left, bottom, width, height]
    # Panel 1: Emagram (Main Sounding)
    ax1 = fig.add_axes([0.1, 0.08, 0.6, 0.85]) 
    # Panel 2: Wind Speed in km/h
    ax2 = fig.add_axes([0.75, 0.08, 0.15, 0.85])

    # 4. Plot Data
    skew.plot(p_hpa, t, 'red', linewidth=3, label='Temperature')
    skew.plot(p_hpa, td, 'green', linewidth=3, label='Dewpoint')
    
    # Place barbs strictly inside the right margin
    skew.plot_barbs(p_hpa[::3], u[::3], v[::3], xloc=1.0)
    # --- PANEL 1: EMAGRAM ---
    # Log scale mimics linear altitude
    ax1.set_yscale('log')
    ax1.set_ylim(1020, 400)
    ax1.set_xlim(-30, 50)

    # 5. Supporting Adiabats
    skew.plot_dry_adiabats(alpha=0.15, color='orangered', linewidth=0.8)
    skew.plot_moist_adiabats(alpha=0.15, color='blue', linewidth=0.8)
    skew.plot_mixing_lines(alpha=0.15, color='green', linestyle=':')
    # Custom Background Lines (Standard Lapse Rate reference)
    # Drawing simple diagonal lines to represent the 'skew'
    for temp in range(-60, 80, 10):
        ax1.plot([temp, temp-40], [1020, 400], color='gray', alpha=0.1, linestyle='-', linewidth=0.5)

    # 6. ALTITUDE LABELS & GRID (Every 0.5 km)
    km_all = np.arange(0, 8.5, 0.5) 
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    # Plot Sounding
    ax1.plot(t, p_hpa, 'red', linewidth=3, label='Temperature')
    ax1.plot(td, p_hpa, 'green', linewidth=3, label='Dewpoint')

    skew.ax.yaxis.set_major_locator(FixedLocator(p_levels))
    skew.ax.yaxis.set_minor_locator(NullLocator()) 
    # Altitude Labels for ax1
    km_all = np.arange(0, 8.5, 0.5)
    p_levels = mpcalc.height_to_pressure_std(km_all * units.km).to(units.hPa).m
    ax1.yaxis.set_major_locator(FixedLocator(p_levels))
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(km_all[pos])} km" if km_all[pos] % 1 == 0 else ""))

    def km_formatter(x, pos):
        if pos < len(km_all):
            val = km_all[pos]
            if val % 1 == 0:
                return f"{int(val)} km"
        return "" 
    ax1.grid(True, which='major', axis='y', color='black', alpha=0.15)
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Altitude (km)", fontsize=12)
    ax1.legend(loc='upper left')

    skew.ax.yaxis.set_major_formatter(FuncFormatter(km_formatter))
    # --- PANEL 2: WIND SPEED ---
    ax2.set_yscale('log')
    ax2.set_ylim(1020, 400)
    ax2.set_xlim(0, 80) # Max 80 km/h wind

    # Vertical grid for easy km reading
    skew.ax.grid(True, which='major', axis='y', color='black', alpha=0.15, linestyle='-')
    # Plot Wind Speed Profile
    ax2.plot(wind_speed_val, p_hpa, 'blue', linewidth=2)
    ax2.fill_betweenx(p_hpa, 0, wind_speed_val, color='blue', alpha=0.1)

    skew.ax.set_ylabel("Altitude (km)")
    skew.ax.set_xlabel("Temperature (°C)")
    # Formatting ax2
    ax2.set_xlabel("Wind (km/h)", fontsize=12)
    ax2.set_yticklabels([]) # Hide altitude labels on the second panel
    ax2.grid(True, which='major', axis='both', alpha=0.2)

    # 7. Metadata and Title
    # Standard labels for wind speed
    ax2.set_xticks([0, 20, 40, 60, 80])
    # 4. Final Polish
    ref_time_str = datetime.datetime.fromisoformat(ds.attrs["ref_time"]).strftime('%Y-%m-%d %H:%M')
    plt.title(f"Paragliding Sounding (Payerne) | {ref_time_str} UTC", fontsize=15, pad=25)
    skew.ax.legend(loc='upper left', frameon=True)
    fig.suptitle(f"Paragliding Emagram & Wind Profile | Payerne | {ref_time_str} UTC", fontsize=16)

    # 8. Save - Explicitly NO tight_layout to preserve our manual adjustments
    plt.savefig("latest_skewt.png", dpi=150)
    print("Success: Tall Summer Sounding generated.")
    print("Success: Custom Emagram with km/h wind panel generated.")

if __name__ == "__main__":
    main()
