# -*- coding: utf-8 -*-
"""

@author: peign

Who can still win the drivers WDC?
======================================

Calculates which drivers still has chance to win the WDC

Simulate win probability based on last race results

See pilot point evolution and estimation over the year for the WDC

"""
#%% Import and initialisation 
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import fastf1
from fastf1.ergast import Ergast
import fastf1.plotting
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False,
                          color_scheme='fastf1')
fastf1.Cache.enable_cache('cache')

import logging
logging.getLogger("fastf1").setLevel(logging.ERROR)

##############################################################################
# For this example, we are looking at the 2024 season.
# We want to know who can theoretically still win the drivers' championship
# after the first 15 races.

SEASON = 2024
ROUND = 19
ROUNDS_USED_FOR_PREDICTION = 5
n_simulations = 500

#if first, 1/2 chance of best lap
points = {'1st': 25.5, '2nd': 18, '3rd': 15, '4th': 12,'5th': 10,'6th': 8,'7th': 6,'8th': 4,'9th': 2,'10th': 1, 'RET': 0}
sprint_points = {'1st': 8, '2nd': 7, '3rd': 6, '4th': 5,'5th': 4,'6th': 3,'7th': 2,'8th': 1,'9th': 0,'10th': 0, 'RET': 0}

#%% Function definition 

##############################################################################
# Get the current driver standings from Ergast.
# Reference https://docs.fastf1.dev/ergast.html#fastf1.ergast.Ergast.get_driver_standings
def get_drivers_standings():
    ergast = Ergast()
    standings = ergast.get_driver_standings(season=SEASON, round=ROUND)
    return standings.content[0]


##############################################################################
# We need a function to calculates the maximum amount of points possible if a
# driver wins everything left of the season.
def calculate_max_points_for_remaining_season():
    POINTS_FOR_SPRINT = 8 + 25 + 1  # Winning the sprint, race and fastest lap
    POINTS_FOR_CONVENTIONAL = 25 + 1  # Winning the race and fastest lap

    events = fastf1.events.get_event_schedule(SEASON, backend='ergast')
    events = events[events['RoundNumber'] > ROUND]
    # Count how many sprints and conventional races are left
    sprint_events = len(events.loc[events["EventFormat"] == "sprint_qualifying"]) #named change in 2024 season, before sprint_shootout
    conventional_events = len(events.loc[events["EventFormat"] == "conventional"])

    # Calculate points for each
    sprint_points = sprint_events * POINTS_FOR_SPRINT
    conventional_points = conventional_events * POINTS_FOR_CONVENTIONAL

    return sprint_points + conventional_points


##############################################################################
# For each driver we will see if there is a chance to get more points than
# the current leader. We assume the leader gets no more points and the
# driver gets the theoretical maximum amount of points.
# This allow to see which driver still as a chance to win WDC

def calculate_who_can_win(driver_standings, max_points, print_data=False):
    LEADER_POINTS = int(driver_standings.loc[0]['points'])

    List_possible_winner={}
    for i, _ in enumerate(driver_standings.iterrows()):
        driver = driver_standings.loc[i]
        driver_max_points = int(driver["points"]) + max_points
        
        if driver_max_points < LEADER_POINTS :
            can_win = 'No' 
        else :
            List_possible_winner[driver['driverCode']]=driver['points']
            can_win = 'Yes'
        if print_data:
            print(f"{driver['position']}: {driver['givenName'] + ' ' + driver['familyName']}, "
                  f"Current points: {driver['points']}, "
                  f"Theoretical max points: {driver_max_points}, "
                  f"Can win: {can_win}")
        
    return List_possible_winner


##############################################################################
# Ths function get each pilot selected theire last position
# These position will be used to estimate prbability of each position for each driver
def get_last_position(drivers,ROUND,ROUNDS_USED_FOR_PREDICTION,SEASON):    
    schedule = fastf1.get_event_schedule(SEASON)
    
    last_races = schedule[schedule['EventFormat'] != 'testing'][max(ROUND-ROUNDS_USED_FOR_PREDICTION,1):ROUND]
    
    Pilot_last_position={}
    for driver_code in drivers:
        Pilot_last_position[driver_code]=[]
    
    for _, race_info in last_races.iterrows():
        session = fastf1.get_session(SEASON, race_info['EventName'], 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)

        race_results = session.results
        for driver_code in drivers:
            driver_result = race_results.loc[race_results['Abbreviation'] == driver_code]
        
            if not driver_result.empty:
                position = driver_result['Position'].values[0]
                Pilot_last_position[driver_code].append(position)
            else:
                Pilot_last_position[driver_code].append(20)
    
    return Pilot_last_position

##############################################################################
# Based on previous result, this function simulated a probability of each driver
# to be at a certain place for the next race

def estimate_position(list_last_position):
    # Compute mean and std of last known position
    mean = np.mean(list_last_position)
    std_dev = np.std(list_last_position)

    #all posible position from 1st to 20th
    possible_positions = np.arange(1, 21)

    # Compute probabilities of position and make a dictionnary
    probabilities = stats.norm(mean, std_dev).pdf(possible_positions)
    probabilities = probabilities / np.sum(probabilities)
    probabilities = {f"{pos}th": round(prob, 4) for pos, prob in zip(possible_positions, probabilities)}
    total_prob = sum(probabilities.values())
    probabilities = {pos: prob / total_prob for pos, prob in probabilities.items()}

    # shorten from 11th to 20th in RET
    ret_prob = sum(probabilities[f'{i}th'] for i in range(11, 21))
    shorten_probabilities = {k: v for k, v in probabilities.items() if k in [f'{i}th' for i in range(1, 11)]}
    shorten_probabilities['RET'] = ret_prob
    final_probabilities = {'1st': shorten_probabilities.pop('1th'),'2nd': shorten_probabilities.pop('2th'),'3rd': shorten_probabilities.pop('3th'),**shorten_probabilities}
    return final_probabilities


##############################################################################
# This function simulate random race event and get each driver a position based on estimated position
# It then sum up all the simulation to give a driver probability to win WDC

def calculate_WDC_probability(List_possible_winner):
    events = fastf1.events.get_event_schedule(SEASON, backend='ergast')
    events = events[events['RoundNumber'] > ROUND]
    # Count how many sprints and conventional races are left
    sprint_events = len(events.loc[events["EventFormat"] == "sprint_qualifying"]) #named change in 2024 season, before sprint_shootout
    conventional_events = len(events.loc[events["EventFormat"] == "conventional"])

    Pilot_tempo_WDC={}
    for pilot in List_possible_winner.keys():
        Pilot_tempo_WDC[pilot]=0
    
    
    for _ in tqdm(range(n_simulations)):
        Pilot_tempo_point={}
        for pilot in List_possible_winner.keys():
            Pilot_tempo_point[pilot]=List_possible_winner[pilot]
        
        for _ in range(conventional_events+sprint_events):
            for pilot in List_possible_winner.keys():
                pilot_finsh=np.random.choice(list(Pilot_probabilities[pilot].keys()), p=list(Pilot_probabilities[pilot].values()))
                Pilot_tempo_point[pilot]+=points[pilot_finsh]
        for _ in range(sprint_events):
                pilot_finsh=np.random.choice(list(Pilot_probabilities[pilot].keys()), p=list(Pilot_probabilities[pilot].values()))
                Pilot_tempo_point[pilot]+=sprint_points[pilot_finsh]
        
        maximum_point = max(Pilot_tempo_point, key=Pilot_tempo_point.get)
        Pilot_tempo_WDC[maximum_point]+=1
    
    WDC_Probability = {key: value*100 / n_simulations for key, value in Pilot_tempo_WDC.items()}
    
    return WDC_Probability

##############################################################################
# This function plot probability of winning for each driver who still have a chance to win WDC

def plot_WDC_probability(WDC_Probability,SEASON,ROUND):

    # Extract keys (drivers) and values (percentages)
    drivers = list(WDC_Probability.keys())
    percentages = list(WDC_Probability.values())
    
    session = fastf1.get_session(SEASON, ROUND, 'R')
    #session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)
    
    team_colors = list()
    for driver in WDC_Probability.keys():
        try : 
            color = fastf1.plotting.get_driver_color(driver, session=session)
            team_colors.append(color)
        except :
            color='#000000'
    
    
    # Create a figure with two axes (to simulate a cut in the x-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1, 1]}, dpi=300)
    
    # Plot the first part: bars with values below 10%
    ax1.barh(drivers, percentages, color=team_colors, edgecolor='grey')
    ax1.set_xlim(0, 30)  # Limit to values below 10%
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Plot the second part: bars with values above 70%
    ax2.barh(drivers, percentages, color=team_colors, edgecolor='grey')
    ax2.set_xlim(50, 100)  # Limit to values above 70%
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Hide the spines between the two plots
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    # Add "cut" marks (slashes) to indicate the break
    d = .015  # Size of the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    
    # Add cut marks at the split (lower end)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # Update transform for the second axis
    ax2.plot((-d, +d), (-d, +d), **kwargs)  
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    
    # Add "cut" marks (slashes) to indicate the break
    # ax1.plot([1, 1], [-0.5, len(drivers) - 0.5], color='k', clip_on=False, transform=ax1.transAxes)
    # ax2.plot([0, 0], [-0.5, len(drivers) - 0.5], color='k', clip_on=False, transform=ax2.transAxes)

    
    # Add labels and title
    fig.text(0.5, 0.04, 'Percentage (%)', ha='center')
    ax1.set_ylabel('Drivers')
    fig.suptitle('Driver probability WDC \n Data extrapolated from: '+ str(session).split(' - ')[0].split(' Season ')[1])
    
    # Show the plot
    plt.subplots_adjust(wspace=0.1)
    plt.show()
    
    return 

##############################################################################
# This function get pilot point evolution through the year

def pilot_point_evolution(SEASON,ROUND):
  
    schedule = fastf1.get_event_schedule(SEASON)    
    races = schedule[schedule['EventFormat'] != 'testing'][:ROUND]
    
    driver_list=[]
    for _, race_info in races.iterrows():
        session = fastf1.get_session(SEASON, race_info['EventName'], 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)
        race_results = session.results
        for driver_code in race_results['Abbreviation']:
            if driver_code not in driver_list:
                driver_list.append(driver_code)
            
        if race_info['EventFormat']=="sprint_qualifying":
            session = fastf1.get_session(SEASON, race_info['EventName'], 'Sprint')
            session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)
            race_results = session.results
            for driver_code in race_results['Abbreviation']:
                if driver_code not in driver_list:
                    driver_list.append(driver_code)
                

    Pilot_point_evolution={}
    
    for _, race_info in races.iterrows():
        
        session = fastf1.get_session(SEASON, race_info['EventName'], 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)
        race_results = session.results
        for driver_code in driver_list:
            if driver_code not in Pilot_point_evolution.keys():
                Pilot_point_evolution[driver_code]=[]
            driver_result = race_results.loc[race_results['Abbreviation'] == driver_code]
            if not driver_result.empty:
                point = driver_result['Points'].values[0]
                Pilot_point_evolution[driver_code].append(point)
            else:
                Pilot_point_evolution[driver_code].append(np.nan)
        
            # If sprint race, need to load sprint results
        if race_info['EventFormat']=="sprint_qualifying":
            session = fastf1.get_session(SEASON, race_info['EventName'], 'Sprint')
            session.load(laps=False, telemetry=False, weather=False, messages=False, livedata=False)
            race_results = session.results
            for driver_code in driver_list:
                
                driver_result = race_results.loc[race_results['Abbreviation'] == driver_code]
                if not driver_result.empty:
                    point = driver_result['Points'].values[0]
                    Pilot_point_evolution[driver_code][-1]+=point
                
    return Pilot_point_evolution

def pilot_cumulative_point(Pilot_point_evolution):
    pilot_cumulative_point_list={}
    for pilot in Pilot_point_evolution.keys():
        list_cumulatie_point=[]
        current_sum = 0
        nan_flag = True  # get if serie start by nan = pilot arrived during season
    
        for point in Pilot_point_evolution[pilot]:
            if np.isnan(point):
                if nan_flag:
                    list_cumulatie_point.append(np.nan)  # if all nan value before, remain nan
                else:
                    list_cumulatie_point.append(current_sum)  # if driver already drive, DNF still count 0 point
            else:
                nan_flag = False  #if a non nan value found, pilot already drive or first drive
                current_sum += point
                list_cumulatie_point.append(current_sum)
        
        pilot_cumulative_point_list[pilot]=list_cumulatie_point

    return pilot_cumulative_point_list 


#%%

def plot_pilot_point_evolution(pilot_cumulative_point_list,SEASON,ROUND,ROUNDS_USED_FOR_PREDICTION):

    session = fastf1.get_session(SEASON, ROUND, 'R')
    schedule = fastf1.get_event_schedule(SEASON)
    max_round = len(schedule[schedule['EventFormat'] != 'testing'])
    
    fig, ax = plt.subplots(figsize=(8.0, 6), dpi=600)

    ##############################################################################
    # For each driver, get their three letter abbreviation (e.g. 'HAM') by simply
    # using the value of the first lap, get their color and then plot their
    # position over the number of laps.
    for pilot in pilot_cumulative_point_list.keys():
        try : 
            style = fastf1.plotting.get_driver_style(identifier=pilot,
                                                      style=['color', 'linestyle'],
                                                      session=session)
        
            ax.plot([i+1 for i in range(ROUND)], pilot_cumulative_point_list[pilot],
                    label=pilot, **style)
        except :
            ax.plot([i+1 for i in range(ROUND)], pilot_cumulative_point_list[pilot],
                    label=pilot, color='black')
    
    for pilot in pilot_cumulative_point_list.keys():
        try : 
            style = fastf1.plotting.get_driver_style(identifier=pilot,
                                                      style=['color'],
                                                      session=session)
            
            current_score=pilot_cumulative_point_list[pilot][-1]
            previous_score_for_estimation=pilot_cumulative_point_list[pilot][-(1+ROUNDS_USED_FOR_PREDICTION)]
            slope=(current_score-previous_score_for_estimation)/ROUNDS_USED_FOR_PREDICTION
            
            
            ax.plot([ROUND,max_round], [current_score,current_score+slope*(max_round-ROUND)],
                    linestyle='dotted', **style)
        except :
            ax.plot([i+1 for i in range(ROUND)], pilot_cumulative_point_list[pilot],
                    color='black',linestyle='dotted')
    # sphinx_gallery_defer_figures
    
    ##############################################################################
    ax.set_xlim([1, max_round])
    ax.set_xlabel('Race')
    ax.set_ylabel('Points')
    ##############################################################################
    # Because this plot is very crowed, add the legend outside the plot area.
    ax.legend(bbox_to_anchor=(1.0, 1.02))
    ax.set_title('Point for each driver \n and estimation of point based on the last '+str(ROUNDS_USED_FOR_PREDICTION)+' races')
    plt.tight_layout()
    
    plt.show()
    
    return 

#%%############################################################################
# Now using the all thoses functions above we can use them to estimate who
# can still win WDC this year.

# Get the current drivers standings
driver_standings = get_drivers_standings()

# Get the maximum amount of points
maximal_points = calculate_max_points_for_remaining_season()

# Print which drivers can still win
List_possible_winner=calculate_who_can_win(driver_standings, maximal_points)

# Get eligible pilot theire last position based on the number of round used for prediction
Pilot_last_position=get_last_position(list(List_possible_winner.keys()),ROUND,ROUNDS_USED_FOR_PREDICTION,SEASON)

# Add some 'bad' position to verstappen to mimic a DNF
Pilot_last_position['VER'].append(18)

# For each pilot, get their probability of race result
Pilot_probabilities = {pilot: estimate_position(Pilot_last_position[pilot]) for pilot in List_possible_winner.keys()}

# Simulate probability of WDC for each pilot
WDC_Probability=calculate_WDC_probability(List_possible_winner)

#Plot the simulation result
plot_WDC_probability(WDC_Probability,SEASON,ROUND)

#%%############################################################################
# Plot point evolution of each pilot throw the year to estimate final result

#get list of pilot and their point trough the season
point_for_pilot=pilot_point_evolution(SEASON,ROUND)

#compute cumulative point for each pilot
list_pilot_cumulative_point=pilot_cumulative_point(point_for_pilot)

#plot point evolution and estimation of futur point based on ROUNDS_USED_FOR_PREDICTION previous race
plot_pilot_point_evolution(list_pilot_cumulative_point,SEASON,ROUND,ROUNDS_USED_FOR_PREDICTION)




