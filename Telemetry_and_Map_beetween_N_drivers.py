# -*- coding: utf-8 -*-
"""
This code generate telemetry which allow you to visualise the selected drivers
You chosse the drivers,the lap or 0 for fasterlap of a session
If only 2 drivers are selected It plot the delta beetween the two drivers
It also create a map and show you the fastest secctor of each driver
"""

# the code is based on the course of https://medium.com/@jaspervhat
# and also to zyxwl2015 who shared her code with me ! thanks again to her !

"""
Needed improvement 
"""

# Try to change color map for cleaner look
# Big improvement : get delta time advance between 2 and plot the gradiant of delta on map (only for 2 drivers) 

"""
Import 
"""

#import library used for this code
#you have to install fastf1 if it's not already done
# -> https://github.com/theOehrly/Fast-F1 for more info
import fastf1 as ff1
from fastf1 import plotting
from fastf1 import utils
from timple.timedelta import strftimedelta
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import pandas as pd
import numpy as np
import os

"""
Data input in order to compare all drivers throught the GP
"""

#list of drivers you want to compare (only the first 3 letters of their names)
drivers=['VER','HAM','NOR']
#drivers=['HAM','VER','PER','BOT','RIC']
#-> another exemple of driver list
race_year=2021
#Name of race or number this year
race_name=20
#race_name='Brasil'
#-> another exemple of race name
#you can use SQ for sprint qualification race or R for Race
type_of_race='R'
#Color map used for the final plot (don't change if the color are good for you)
color_map_plt='hot'
#color_map_plt='ocean'
#Select the lap you want to visualize - if you want fastest lap, put 0
Lap_analyse = 0
#if you want to save the plot put True else put False
save=True

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

# Enable the cache - you have to create a folder name cache on the same repertory as your code
ff1.Cache.enable_cache('cache')

# Load the session data
race = ff1.get_session(race_year, race_name, type_of_race)

# Collect all race laps
laps = race.load_laps(with_telemetry=True)

# Calculate RaceLapNumber (LapNumber minus 1 since the warmup lap is included in LapNumber)
laps['RaceLapNumber'] = laps['LapNumber'] - 1


max_lap = max(laps['RaceLapNumber'])
if Lap_analyse > max_lap:
    print('\n')
    print('---------------------------------------------')
    print('The selected Lap if greater than the last lap')
    print('---------------------------------------------')
    print('\n')
    exit()


#for all drivers get the data and telemetry
list_telemetry=[]
list_legend=[]
#only used if compare 2 drivers :
lap_list_for_delta=[]
for driver in drivers:
    legend_tempo=str(driver)+' '
    # Get laps of the drivers
    tempo_driver_laps=laps.pick_driver(driver)

    
    # Get telemetry from fastest laps or selected lap
    if Lap_analyse == 0:
        tempo_selected_lap = tempo_driver_laps.pick_fastest()
    else:
        for lap in laps.pick_driver(driver).iterlaps():
            if lap[1]['LapNumber']==Lap_analyse:
                tempo_selected_lap= lap[1]
    
    legend_tempo+=strftimedelta(tempo_selected_lap['LapTime'], '%m:%s.%ms')+' '+tempo_selected_lap['Compound']

    #add Lap number if fastest lap
    if Lap_analyse==0:
        legend_tempo+=" - Lap : " + str(tempo_selected_lap['LapNumber'])

    lap_list_for_delta.append(tempo_selected_lap)
    list_legend.append(legend_tempo)
    list_telemetry.append(tempo_selected_lap.get_car_data().add_distance())

"""
Prepare the plot for 2 drivers or for another number
"""

#if 2 drivers compared add delta beetwen the two, else no delta
if len(drivers)==2:
    fig = plt.figure(figsize=(30, 15), dpi=300)
    fig_shape=(11,3)
    ax=[0,0,0,0,0,0,0,0]
    ax[0] = plt.subplot2grid(fig_shape, (0,0), colspan=2, rowspan=5) # Speed
    ax[1] = plt.subplot2grid(fig_shape, (5,0), colspan=2)            # Throttle
    ax[2] = plt.subplot2grid(fig_shape, (6,0), colspan=2)            # Brake
    ax[3] = plt.subplot2grid(fig_shape, (7,0), colspan=2, rowspan=2) # RPM
    ax[4] = plt.subplot2grid(fig_shape, (9,0), colspan=2)            # Gear
    ax[5] = plt.subplot2grid(fig_shape, (10,0), colspan=2)           # Delta
    ax[6] = plt.subplot2grid(fig_shape, (0,2), rowspan=2)            # Legend
    ax[7] = plt.subplot2grid(fig_shape, (2,2), rowspan=11)           # Map

    #plot delta beetween the 2 drivers
    delta_time, ref_tel, compare_tel = utils.delta_time(lap_list_for_delta[0], lap_list_for_delta[1])
    ax[5].plot(ref_tel['Distance'], delta_time, '-', color='white')
    ax[5].axhline(y=0, linestyle='--', color='white', linewidth=0.5)
    ax[5].set_ylabel('Delta', fontsize=label_size)
    ax[5].set_xlim([0,max(list_telemetry[0]['Distance'])])


else :
    fig = plt.figure(figsize=(30, 15), dpi=300)
    fig_shape=(10,3)
    ax=[0,0,0,0,0,0,0]
    ax[0] = plt.subplot2grid(fig_shape, (0,0), colspan=2, rowspan=5) # Speed
    ax[1] = plt.subplot2grid(fig_shape, (5,0), colspan=2)            # Throttle
    ax[2] = plt.subplot2grid(fig_shape, (6,0), colspan=2)            # Brake
    ax[3] = plt.subplot2grid(fig_shape, (7,0), colspan=2, rowspan=2) # RPM
    ax[4] = plt.subplot2grid(fig_shape, (9,0), colspan=2)            # Gear
    ax[5] = plt.subplot2grid(fig_shape, (0,2), rowspan=3)            # Legend
    ax[6] = plt.subplot2grid(fig_shape, (3,2), rowspan=10)           # Map


"""
Difference suptitle if fastest lap or selected lap
"""   

if Lap_analyse==0:
    fig.suptitle("Fastest Lap Telemetry Comparison at "+race.weekend.name+" "+str(race.weekend.year)+" - "+str(type_of_race), fontsize=Title_size)

else :
    fig.suptitle("Lap "+str(Lap_analyse)+" Telemetry Comparison at "+race.weekend.name+" "+str(race.weekend.year)+" - "+str(type_of_race), fontsize=Title_size)

"""
Plot telemetry for all drivers
"""   

#Plot all telemetry data for each driver
for i in range(len(drivers)):
    linestyle,color=data_pilot_color[drivers[i]]
    label=drivers[i]+" "+"insert time"
    ax[0].plot(list_telemetry[i]['Distance'], list_telemetry[i]['Speed'], linestyle=linestyle, color=color, linewidth=line_width)
    #plot Throttle via telemetry  
    ax[1].plot(list_telemetry[i]['Distance'], list_telemetry[i]['Throttle'], linestyle=linestyle, color=color, linewidth=line_width)
    #plot Brake via telemetry
    ax[2].plot(list_telemetry[i]['Distance'], list_telemetry[i]['Brake'],linestyle=linestyle, color=color, linewidth=line_width)
    #plot RPM via telemetry
    ax[3].plot(list_telemetry[i]['Distance'], list_telemetry[i]['RPM'],linestyle=linestyle, color=color, linewidth=line_width)            
    #plot Gear via telemetry
    ax[4].plot(list_telemetry[i]['Distance'], list_telemetry[i]['nGear'], linestyle=linestyle, color=color, linewidth=line_width)
    #fake plot to get legend
    ax[-2].plot([0,0], [0,0], linestyle=linestyle, color=color,label=list_legend[i])


"""
Plot Map on ax[-1]
""" 

def generate_telemetry_for_map_plot_specific_lap(drivers, laps, Lap_analyse):

    # Get rid of some pandas warnings that are not relevant for us at the moment
    pd.options.mode.chained_assignment = None
    telemetry = pd.DataFrame()
    
    # Telemetry can only be retrieved driver-by-driver
    for driver in drivers:
        # Since we want to compare distances, we need to collect telemetry lap-by-lap to reset the distance
    
        for lap in laps.pick_driver(driver).iterlaps():
            #need to investigqte why there a difference of 1 lap
            if lap[1]['LapNumber']==Lap_analyse+1:
                driver_telemetry = lap[1].get_telemetry().add_distance()
                driver_telemetry['Driver'] = driver
                driver_telemetry['Lap'] = lap[1]['RaceLapNumber']
                driver_telemetry['Compound'] = lap[1]['Compound']
                telemetry = telemetry.append(driver_telemetry)
            
    # Only keep required columns
    telemetry = telemetry[['Lap', 'Distance', 'Speed','Driver', 'X','Y']]
    
    # We want 100 mini-sectors
    num_minisectors = 100
    
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
        
    return telemetry

def generate_telemetry_for_map_plot_fastest_lap(drivers, laps):

    # Get rid of some pandas warnings that are not relevant for us at the moment
    pd.options.mode.chained_assignment = None
    telemetry = pd.DataFrame()
    
    # Telemetry can only be retrieved driver-by-driver
    for driver in drivers:
        # Since we want to compare distances, we need to collect telemetry lap-by-lap to reset the distance
        driver_lap=laps.pick_driver(driver).pick_fastest()
        driver_telemetry = driver_lap.get_telemetry().add_distance()
        driver_telemetry['Driver'] = driver
        driver_telemetry['Lap'] = driver_lap['RaceLapNumber']
        driver_telemetry['Compound'] = driver_lap['Compound']
        telemetry = telemetry.append(driver_telemetry)
                
            
    # Only keep required columns
    telemetry = telemetry[['Distance', 'Speed','Driver', 'X','Y']]
    
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
    average_speed = telemetry.groupby([ 'Minisector', 'Driver'])['Speed'].mean().reset_index()
    
    # Select the driver with the highest average speed
    fastest_drivers = average_speed.loc[average_speed.groupby([ 'Minisector'])['Speed'].idxmax()]
    
    # Get rid of the speed column and rename the driver column
    fastest_drivers = fastest_drivers[[ 'Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_driver'})
    
    # Join the fastest driver per minisector with the full telemetry
    telemetry = telemetry.merge(fastest_drivers, on=['Minisector'])
    
    # Order the data by distance to make matploblib does not get confused
    telemetry = telemetry.sort_values(by=['Distance'])
    
    # Assign integer value to the driver because that's what matplotlib wants
    for i in range(len(drivers)):
        telemetry.loc[telemetry['Fastest_driver'] == drivers[i], 'Fastest_driver_int'] = i+1
        
    return telemetry

#plot the race with all the minisector     
def generate_minisector_plot(ax, label_size, telemetry, details=True):

    x = np.array(telemetry['X'].values)
    y = np.array(telemetry['Y'].values)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    driver = telemetry['Fastest_driver_int'].to_numpy().astype(float)


    #need to change color_map_plt for cleaner look
    cmap = cm.get_cmap(color_map_plt, len(drivers))
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(driver)
    lc_comp.set_linewidth(2)

    ax[-1] = plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.xlabel('Lap fastest sector comparaison', fontsize=label_size)
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    
    if details:
        cbar = plt.colorbar(mappable=lc_comp, boundaries=np.arange(1, len(drivers)+2),aspect=100)
        cbar.set_ticks(np.arange(1.5, 9.5))
        cbar.set_ticklabels(drivers)
        #bigger driver name nex to color bar
        cbar.ax.tick_params(labelsize=0.75*label_size)

if Lap_analyse==0:
    telemetry = generate_telemetry_for_map_plot_fastest_lap(drivers, laps)
else :
    telemetry = generate_telemetry_for_map_plot_specific_lap(drivers, laps, Lap_analyse)

generate_minisector_plot(ax,label_size, telemetry, details=True)


"""
Add legend, titles etc.
"""  

# Set all labels
ax[0].set_ylabel('Speed [km/h]', fontsize=label_size)
ax[1].set_ylabel('Throttle', fontsize=label_size)
ax[1].set_ylim([-1,101])
ax[2].set_ylabel('Brake', fontsize=label_size)
ax[2].set_ylim([-1,101])
ax[3].set_ylabel('RPM', fontsize=label_size)
ax[4].set_ylabel('Gear', fontsize=label_size)
ax[-3].set_xlabel('Distance [m]', fontsize=label_size)
ax[-2].legend(fontsize=label_size, loc='upper left')
ax[-2].get_xaxis().set_visible(False)
ax[-2].get_yaxis().set_visible(False)
ax[4].set_yticks([0,4,8])
ax[0].set_xlim([-5,max(list_telemetry[0]['Distance'])+5])
ax[1].set_xlim([-5,max(list_telemetry[0]['Distance'])+5])
ax[2].set_xlim([-5,max(list_telemetry[0]['Distance'])+5])
ax[3].set_xlim([-5,max(list_telemetry[0]['Distance'])+5])
ax[4].set_xlim([-5,max(list_telemetry[0]['Distance'])+5])
for i in ax[0:-2]:
    i.grid()
    
plt.tight_layout()    
fig.subplots_adjust(hspace = 0.0)

plt.xlim(0,max(list_telemetry[0]['Distance']))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

"""
Save plot
"""

#Save plot
repository ='img/'+str(race.weekend.year)+"_"+race.weekend.name
def save_plot(repertorie):
    if not os.path.exists(repository):
        os.makedirs(repository)
    
    str_driver = drivers[0]
    for i in drivers[1:]:
        str_driver+=" vs "+i
    
    plt_save_name=repository+"/Telemetry_"+type_of_race+"_"+str_driver
    
    if Lap_analyse==0:
        plt_save_name+=" Fastest Lap"
    else :
        plt_save_name+="_Lap_"+str(Lap_analyse)
    plt.savefig(plt_save_name+".png", dpi=300)
    
if save :
    save_plot(repository)

plt.show()


#that all for now
#version of 04/12/2021 by Teinge
#feel free to share as long as you reference my work
