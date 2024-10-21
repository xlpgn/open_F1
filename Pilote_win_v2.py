# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:47:32 2024

@author: peign
"""
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

n_simulations = 50000


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


Pilot_list=['Lando','Max','Leclerc', 'Bottas']
Pilot_last_position={'Lando': [4, 1, 4, 3, 1], 
                     'Max' :  [3, 2, 5, 6, 2, 15], 
                     'Leclerc' : [1, 5, 2, 1, 3],
                     'Bottas' : [17, 16, 16, 16,15]}

Pilot_points={'Lando': 297, 
              'Max' :  354, 
              'Leclerc' : 275,
              'Bottas' : 0}

Pilot_probabilities = {pilot: estimate_position(Pilot_last_position[pilot]) for pilot in Pilot_list}


#if first, 1/2 chance of best lap
points = {'1st': 25.5, '2nd': 18, '3rd': 15, '4th': 12,'5th': 10,'6th': 8,'7th': 6,'8th': 4,'9th': 2,'10th': 1, 'RET': 0}
sprint_points = {'1st': 8, '2nd': 7, '3rd': 6, '4th': 5,'5th': 4,'6th': 3,'7th': 2,'8th': 1,'9th': 0,'10th': 0, 'RET': 0}

# remaining races
n_races = 5
# remaining sprint races
n_races_sprint = 2

Pilot_tempo_WDC={}
for pilot in Pilot_list:
    Pilot_tempo_WDC[pilot]=0


for _ in tqdm(range(n_simulations)):
    Pilot_tempo_point={}
    for pilot in Pilot_list:
        Pilot_tempo_point[pilot]=Pilot_points[pilot]
    
    for _ in range(n_races):
        for pilot in Pilot_list:
            pilot_finsh=np.random.choice(list(Pilot_probabilities[pilot].keys()), p=list(Pilot_probabilities[pilot].values()))
            Pilot_tempo_point[pilot]+=points[pilot_finsh]
    for _ in range(n_races_sprint):
            pilot_finsh=np.random.choice(list(Pilot_probabilities[pilot].keys()), p=list(Pilot_probabilities[pilot].values()))
            Pilot_tempo_point[pilot]+=sprint_points[pilot_finsh]
    
    maximum_point = max(Pilot_tempo_point, key=Pilot_tempo_point.get)
    Pilot_tempo_WDC[maximum_point]+=1

WDC_Probability = {key: value*100 / n_simulations for key, value in Pilot_tempo_WDC.items()}

# lando_win_probability = lando_wins / n_simulations
# max_win_probability = max_win / n_simulations
# leclerc_win_probability = leclerc_win / n_simulations

# max_mean_point=max_mean_point/n_simulations
# lando_mean_point=lando_mean_point/n_simulations
# leclerc_mean_point=leclerc_mean_point/n_simulations

print('-------')
for pilot in Pilot_list:
    print("Probability of ", pilot, " winning: ", round(WDC_Probability[pilot],2),' %')
print('-------')


