"""
This code generate a plot which allow you to visualise N out of 20 drivers during a race. 
You choose a reference drivers and you can then visualise the gap between this drivers and the others
"""

# the code is inspired of the work of ProjectF1

"""
Needed improvement 
"""

# have some trouble with some race, don't really know why
#have trouble to get the best N drivers, sometime got drivers that DNF, strange

"""
Import 
"""

#import library used for this code
#you have to install fastf1 if it's not already done
# -> https://github.com/theOehrly/Fast-F1 for more info
import fastf1 as ff1
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import pandas as pd
import os

"""
Data input in order to compare all drivers throught the GP
"""

#select the driver you want as a reference (only the 3 letter of the name : Ham for Hamilton)
DriverREF='LEC'
race_year=2021
#Name of race or number this year
race_name=19
#race_name='Brasil'
#-> another exemple of race name
#you can use SQ for sprint qualification race or R for Race
type_of_race='R'
#Color map used for the final plot (don't change if the color are good for you)
number_driver_wanted=12
#if you want to save the plot put True else put False
save=True
#put true if want comment on plot
plot_comment=True


"""
Manual selection of drivers
"""

#all_pilot_list=['HAM','BOT','SAI','LEC','VER','PER','RIC','NOR','VET','STR','ALO','OCO','TSU','GAS','MSC','MAZ','RAI','GIO','LAT','RUS']
manual_drivers_list=['HAM','BOT','SAI','LEC','GAS','VER','ALO','OCO','NOR']
use_manual_driver=True

"""
Comment Setup
"""

#List_comment = [[Comment,xypos of event,xypos of comment,xlength elipse,ylength elipse, angle elipse]]
#round ratio 3/10
List_comment = [['NOR saved by SC',(8,0),(3,20),3,10,0],
                ['HAM strong undercut',(26,10),(18,40),4,20,-10],
                ['GAS overcomes alpine',(58,-18),(50,-40),3,10,0],
                ['Ferrari make gap',(60,-2),(55,20),20,20,0]]



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
laps = race.load_laps(with_telemetry=False)

# Calculate RaceLapNumber (LapNumber minus 1 since the warmup lap is included in LapNumber)
laps['RaceLapNumber'] = laps['LapNumber'] -1
laps=laps[laps['RaceLapNumber']!=0]

"""
Usefull fonction
"""
#Function to get a list of N driver 
# -> need to be change to get only the N first drivers


def get_list_n_drivers(n):
    List_driver_final=[]
    for driver_result in race.results:
        List_driver_final.append(driver_result['Driver']['code'])
    return List_driver_final[0:n]


#get a list of N driver with function or use manual list
if use_manual_driver:
    Drivers_list=manual_drivers_list
else :
    Drivers_list=get_list_n_drivers(number_driver_wanted)



#Function that return a list of cumulative time of a drivers throught the race
def get_cumulative_time(driver_name):
    laps_drivers = laps.pick_driver(driver_name)
    time_laps=laps_drivers['LapTime'].tolist()
    time_start=laps_drivers['LapStartTime'].tolist()[0]
    cumulative_time_laps=[time_laps[0]+time_start]
    for j in time_laps[1:]:
        cumulative_time_laps.append(cumulative_time_laps[-1]+j)
    return [driver_name,cumulative_time_laps]


#get the data of the reference  in order to compare with all the other drivers
laps_drivers_ref = laps.pick_driver(DriverREF)
Driver_ref_data=get_cumulative_time(DriverREF)


#Function to create a list of gap at each lap beetween a driver and the ref driver
def difference_driver_with_ref(list_A,list_Ref=Driver_ref_data[1]):
    len_list=len(list_A)
    return [(list_Ref[j]-list_A[j]).total_seconds() for j in range(len_list)]


"""
Plot
"""

#Configuration of the plot
#figure(figsize=(32, 20), dpi=300)
#fig, ax = plt.subplots()

fig, ax = plt.subplots(1,1,figsize=(32,20))


#plot each drivers and add the plot option and color of each driver
for driver in Drivers_list:
    data=get_cumulative_time(driver)
    linestyle,color=data_pilot_color[data[0]]
    data=difference_driver_with_ref(data[1])
    plt.plot(laps_drivers_ref['RaceLapNumber'][0:len(data)],data, label=driver, linestyle=linestyle, color=color, linewidth=line_width)


"""
Add comment on the plot
"""

#round ratio 3/10 between x and y
if plot_comment :
    for comment in List_comment:
        #comment, xy localisation, xy text localisation
        
        
        #plot ellipse around comment zone
        #xy localisation of center, xwidth, ywidth, angle
        el=Ellipse(comment[1], comment[3], comment[4], angle=comment[5], 
                              edgecolor='white', linestyle=':', linewidth =6,facecolor ='none', alpha=0.8)
        ax.add_artist(el)
        
        

        ax.annotate(comment[0], xy=comment[1], xytext=comment[2],size=label_size,
                    arrowprops=dict(arrowstyle="->",patchB=el, color='white',linewidth=label_size/6));
                            #connectionstyle="angle3,angleA=0,angleB=-90"));
"""
Plot options
"""

#plot from lap 1 to the last lap
plt.xlim(1,laps_drivers_ref['RaceLapNumber'].tolist()[-1])
# plt.ylim(-5,30) 
#-> can be used to restaint the y axes 
plt.ylabel('Gap (Seconds)', fontsize=label_size)
plt.xlabel('Lap', fontsize=label_size)

plt.title("Gap to "+ DriverREF+" at "+str(race.weekend.year)+" "+race.weekend.name, fontsize=Title_size)
#plt.legend(bbox_to_anchor=(0., -0.25, 1., .0), loc='lower left',ncol=10, mode="expand", borderaxespad=0.)
plt.legend(bbox_to_anchor=(0., 0, 1., .0), loc='lower left',ncol=10, mode="expand", borderaxespad=0., fontsize=label_size)


# add a thin grid view, can be change for a more precise one, i have to look that
#-> can be change
plt.grid(alpha=0.3)

plt.xticks(fontsize=30)
plt.yticks(fontsize=30)


"""
Save
"""

#save the plot if needed
if save:
    repertorie='img/'+str(race.weekend.year)+"_"+race.weekend.name
    if not os.path.exists(repertorie):
        os.makedirs(repertorie)
    if plot_comment:
        plt.savefig(repertorie+"/Gap_to_"+DriverREF+"_"+type_of_race+"_commented.png", dpi=300,bbox_inches='tight')
    else :
        plt.savefig(repertorie+"/Gap_to_"+DriverREF+"_"+type_of_race+".png", dpi=300,bbox_inches='tight')

plt.show()


#that all for now
#version of 02/12/2021 by Teinge
#feel free to share as long as you reference my work
