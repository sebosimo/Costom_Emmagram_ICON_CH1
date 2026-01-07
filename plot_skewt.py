import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from metpy.units import units
import metpy.calc as mpcalc
from metpy.plots import SkewT

def main():
    # ... (Keep your data loading logic here) ...
    # Assuming p, t, td, u, v are already extracted as Pint quantities
    
    # 1. Unit Conversions (Match the Swiss Plot)
    p_hpa = p.to('hPa')
    t_c = t.to('degC')
    td_c = td.to('degC')
    wind_speed_kt = mpcalc.wind_speed(u, v).to('knots')
    
    # 2. Setup Figure and Skew-T Grid
    fig = plt.figure(figsize=(12, 9))
    # The SkewT class creates the main sounding area
    skew = SkewT(fig, rotation=45) 
    ax1 = skew.ax
    
    # Set Limits
    ax1.set_ylim(1050, 300)
    ax1.set_xlim(-20, 50)
    
    # 3. Add the "Background" Meteorological Lines
    skew.plot_dry_adiabats(colors='gray', alpha=0.25, linewidth=1)
    skew.plot_moist_adiabats(colors='gray', alpha=0.25, linewidth=1)
    skew.plot_mixing_lines(colors='gray', alpha=0.2, linestyle='dotted')
    
    # 4. Plot Temperature and Dewpoint
    # (Use black for 'Zurich' style or green for 'Payerne')
    skew.plot(p_hpa, t_c, 'black', linewidth=2, label='Temperature')
    skew.plot(p_hpa, td_c, 'black', linestyle='--', linewidth=2, label='Dewpoint')

    # 5. Add Wind Barbs (Positioned on the right side of the main plot)
    # Filter data to avoid cluttered barbs (every 50hPa)
    interval = np.where(p_hpa.m % 50 == 0)
    skew.plot_barbs(p_hpa[interval], u[interval], v[interval], 
                   xloc=1.05, color='black', length=6)

    # 6. Wind Speed Panel (Right side)
    # Create a new axes for wind speed profile
    ax2 = fig.add_axes([0.78, 0.11, 0.15, 0.77])
    ax2.plot(wind_speed_kt, p_hpa, color='black', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_ylim(1050, 300)
    ax2.set_xlim(0, 50)
    ax2.set_xlabel("Windspeed (kt)")
    ax2.grid(True)

    # 7. Add Height Labels (Right side of Skew-T)
    # This maps pressure to standard altitude labels like in the image
    ax1.set_ylabel("Pressure (hPa)")
    ax1.set_xlabel("Temperature (°C)")
    
    plt.show()

if __name__ == "__main__":
    main()
