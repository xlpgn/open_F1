# -*- coding: utf-8 -*-
"""
This code generate a plot of the track which allow you to visualise the fastest drivers by mini-sector
You can choose the drivers, the race and the lap you want to see
"""

# the code is based on the course of https://medium.com/@jaspervhat
# and the word of ProjectF1

"""
Needed improvement 
"""

# The improvement could be to have a more precise comparaison of time
#if we select only 2 drivers

"""
Import 
"""

#import library used for this code
#you have to install fastf1 if it's not already done
# -> https://github.com/theOehrly/Fast-F1 for more info

import fastf1 as ff1
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import numpy as np
import pandas as pd
import os

"""
Data input in order to compare 2 or more drivers over the GP
"""

#list of drivers you want to compare (only the first 3 letters of their names)
drivers=['BOT','HAM','VER','PER']
#drivers=['HAM','VER']
#-> another exemple of driver list
race_year=2021
#Name of race or number this year
#race_name='Brasil'
race_name=20
#you can use SQ for sprint qualification race or R for Race
type_of_race='R'
#Color map used for the final plot (don't change if the color are good for you)
color_map_plt='bwr'
#Select the first and last lap you want to visualize
#if the lap max you choose is greater than the last lap of the race the code will correct it
Lap_min = 50
Lap_max = 57


"""
Plotting setup
"""

# Setup plotting
plotting.setup_mpl()

line_width = 2.0
label_size = 30
Title_size = 1.5*label_size

#get the collor of all drivers and data for the plot option
#fast f1 have function for that, but i prefere doing like that
data_pilot_color={'HAM':('-','#00d2be'),'BOT':('--','#00d2be'),'VER':('-','#0600ef'),'PER':('--','#0600ef'),
                  'NOR':('--','#ff8700'),'RIC':('-','#ff8700'),'LEC':('-','#dc0000'),'SAI':('--','#dc0000'),
                  'GAS':('-','#2b4562'),'TSU':('--','#2b4562'),'OCO':('-','#0090ff'),'ALO':('--','#0090ff'),
                  'VET':('-','#006f62'),'STR':('--','#006f62'),'LAT':('-','#005aff'),'RUS':('--','#005aff'),
                  'MSC':('-','#ffffff'),'MAZ':('--','#ffffff'),'RAI':('-','#900000'),'GIO':('--','#900000')}



"""
Code
"""

ff1.Cache.enable_cache('cache')
# Get rid of some pandas warnings that are not relevant for us at the moment
pd.options.mode.chained_assignment = None

# Load the session data
race = ff1.get_session(race_year, race_name, type_of_race)

# Get the laps
laps = race.load_laps(with_telemetry=True)

# Calculate RaceLapNumber (LapNumber minus 1 since the warmup lap is included in LapNumber)
laps['RaceLapNumber'] = laps['LapNumber'] # -1
max_lap = max(laps['RaceLapNumber'])
#get max lap beetween the one choosen and the last lap of the race
Lap_max=int(min(max_lap,Lap_max))
laps = laps.loc[laps['RaceLapNumber'] >= Lap_min]


telemetry = pd.DataFrame()

# Telemetry can only be retrieved driver-by-driver
for driver in drivers:
    driver_laps = laps.pick_driver(driver)
    # Since we want to compare distances, we need to collect telemetry lap-by-lap to reset the distance
    for lap in driver_laps.iterlaps():
        driver_telemetry = lap[1].get_telemetry().add_distance()
        driver_telemetry['Driver'] = driver
        driver_telemetry['Lap'] = lap[1]['RaceLapNumber']
        driver_telemetry['Compound'] = lap[1]['Compound']
    
        telemetry = telemetry.append(driver_telemetry)
        
# Only keep required columns
telemetry = telemetry[['Lap', 'Distance', 'Speed','Driver', 'X','Y']]

# We want 50 mini-sectors
num_minisectors = 50

# What is the total distance of a lap?
total_distance = max(telemetry['Distance'])

# Generate equally sized mini-sectors 
minisector_length = total_distance / num_minisectors

minisectors = [0]

for i in range(0, (num_minisectors - 1)):
    minisectors.append(minisector_length * (i + 1))
    
# Assign minisector to every row in the telemetry data
telemetry['Minisector'] =  telemetry['Distance'].apply(
  lambda z: (
    minisectors.index(
      min(minisectors, key=lambda x: abs(x-z)))+1
  )
)

# Calculate mean speed for each drivers by mini-sector
average_speed = telemetry.groupby(['Lap', 'Minisector', 'Driver'])['Speed'].mean().reset_index()


# Select the driver with the highest average speed
fastest_drivers = average_speed.loc[average_speed.groupby(['Lap', 'Minisector'])['Speed'].idxmax()]

# Get rid of the speed column and rename the driver column
fastest_drivers = fastest_drivers[['Lap', 'Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_driver'})

# Join the fastest driver per minisector with the full telemetry
telemetry = telemetry.merge(fastest_drivers, on=['Lap', 'Minisector'])

# Order the data by distance to make matploblib does not get confused
telemetry = telemetry.sort_values(by=['Distance'])

# Assign integer value to the driver because that's what matplotlib wants
for i in range(len(drivers)):
    telemetry.loc[telemetry['Fastest_driver'] == drivers[i], 'Fastest_driver_int'] = i+1

#get the fastest drivers beetween the drivers choosen for a lap
def fastest_driver_lap(lap):
    data_faster_driver = laps[['LapTime', 'RaceLapNumber','Driver']]
    time_all_drivers=[]
    for i in drivers :
        time_driver_laps = data_faster_driver[data_faster_driver['Driver']==i]
        try :
            time_driver_lap =  time_driver_laps[time_driver_laps['RaceLapNumber']==lap]['LapTime'].tolist()[0].total_seconds()
        except:
            time_driver_lap=10000
        time_all_drivers.append(time_driver_lap)
    index_min=time_all_drivers.index(min(time_all_drivers))
    second_time_list=np.copy(time_all_drivers)
    second_time_list[index_min]=100000
    return(round(min(second_time_list)-min(time_all_drivers),3),drivers[index_min])

#plot the race with all the minisector     
def generate_minisector_plot(lap, save=False, details=True):
    single_lap = telemetry.loc[telemetry['Lap'] == lap]

    x = np.array(single_lap['X'].values)
    y = np.array(single_lap['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    driver = single_lap['Fastest_driver_int'].to_numpy().astype(float)

    cmap = cm.get_cmap(color_map_plt, len(drivers))
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(driver)
    lc_comp.set_linewidth(2)

    plt.rcParams['figure.figsize'] = [12, 5]
    
    str_driver = drivers[0]
    for i in drivers[1:]:
        str_driver+=" vs "+i
    
    fastest_lap_dif,fastest_driver=fastest_driver_lap(lap)
    
    if details:
        plt.suptitle(
            str(race.weekend.year)+" "+race.weekend.name+" \n"+str_driver+" | Lap = "+str(lap)
            +" \n"+fastest_driver+" is faster by "+str(fastest_lap_dif)
        )
        
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    
    if details:
        cbar = plt.colorbar(mappable=lc_comp, boundaries=np.arange(1, len(drivers)+2))
        cbar.set_ticks(np.arange(1.5, 9.5))
        cbar.set_ticklabels(drivers)
    
    if save:
        repertorie='img/'+str(race.weekend.year)+"_"+race.weekend.name
        if not os.path.exists(repertorie):
            os.makedirs(repertorie)
        plt.savefig(repertorie+"/Minisectors_lap_"+str(lap)+"_"+str_driver+".png", dpi=300)

    plt.show()


for i in range(Lap_min,Lap_max+1):
    generate_minisector_plot(i, save=True, details=True)

#that all for now
#version of 26/11/2021 by Teinge
#feel free to share as long as you reference my work