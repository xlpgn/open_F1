# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:47:32 2024

@author: peign
"""
import numpy as np
import scipy.stats as stats

n_simulations = 10000

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


max_last_positions = [5, 2, 5, 4, 2, 6, 5, 2,12]
#max_probabilities = {'3rd': 0.10, '4th-6th': 0.8375, 'RET': 0.0625}
max_probabilities = estimate_position(max_last_positions)


Lando_last_positions = [3, 2, 5, 1, 3, 4, 1]
#lando_probabilities = {'1st': 0.2875, '2nd': 0.3125, '4th': 0.3375, 'RET': 0.0625}
lando_probabilities = estimate_position(Lando_last_positions)


#if first, 1/2 chance of best lap
points = {'1st': 25.5, '2nd': 18, '3rd': 15, '4th': 12,'5th': 10,'6th': 8,'7th': 6,'8th': 4,'9th': 2,'10th': 1, 'RET': 0}
sprint_points = {'1st': 8, '2nd': 7, '3rd': 6, '4th': 5,'5th': 4,'6th': 3,'7th': 2,'8th': 1,'9th': 0,'10th': 0, 'RET': 0}


# current points after Singapour
max_initial_points = 331
lando_initial_points = 279
# remaining races
n_races = 6
# remaining sprint races
n_races_sprint = 3

#Variable
lando_wins = 0
max_mean_point=0
lando_mean_point=0

for _ in range(n_simulations):
    max_points = max_initial_points
    lando_points = lando_initial_points

    for _ in range(n_races):
        max_finish = np.random.choice(list(max_probabilities.keys()), p=list(max_probabilities.values()))
        lando_finish = np.random.choice(list(lando_probabilities.keys()), p=list(lando_probabilities.values()))
        max_points += points[max_finish]
        lando_points += points[lando_finish]
    
    for _ in range(n_races_sprint):
        max_finish = np.random.choice(list(max_probabilities.keys()), p=list(max_probabilities.values()))
        lando_finish = np.random.choice(list(lando_probabilities.keys()), p=list(lando_probabilities.values()))
        max_points += sprint_points[max_finish]
        lando_points += sprint_points[lando_finish]
    
    max_mean_point+=max_points
    lando_mean_point+=lando_points
    
    if lando_points > max_points:
        lando_wins += 1

lando_win_probability = lando_wins / n_simulations
max_mean_point=max_mean_point/n_simulations
lando_mean_point=lando_mean_point/n_simulations
print("Probability of Lando winning: ", round(lando_win_probability * 100,2))
print("Point : ", int(max_mean_point), ' vs ', int(lando_mean_point))

