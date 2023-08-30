import os
import json
import plotly
import warnings
import numpy as np

from tqdm import tqdm
from scipy.stats import poisson
from scipy.optimize import minimize

from utils.functions import *

warnings.filterwarnings('ignore')
plotly.offline.init_notebook_mode()

def bp_cdf(x1, x2, l1, l2, theta):
    i = np.arange(x1 + 1)
    j = np.arange(x2 + 1)
    m1 = np.minimum.outer(poisson.cdf(i, theta * l1), poisson.cdf(j, theta * l2))
    m2 = np.multiply.outer(poisson.pmf(x1 - i, (1 - theta) * l1), poisson.pmf(x2 - j, (1 - theta) * l2))
    return (m1 * m2).sum()

def bp_pmf(x1, x2, l1, l2, theta):
    return bp_cdf(x1, x2, l1, l2, theta) \
            - bp_cdf(x1 - 1, x2, l1, l2, theta) \
            - bp_cdf(x1, x2 - 1, l1, l2, theta) \
            + bp_cdf(x1 - 1, x2 - 1, l1, l2, theta)

def bp_logpmf(x1, x2, l1, l2, theta):
    return np.log(bp_pmf(x1, x2, l1, l2, theta))

def bp_likelihood(parameters, played_games, inx):
    lik = 0
    for home in played_games:
        for away in played_games[home]:
            result = played_games[home][away]
            l1 = parameters[inx[home]['Atk']] / parameters[inx[away]['Def']]
            l2 = parameters[inx[away]['Atk']] / parameters[inx[home]['Def']]
            l3 = parameters[inx[home]['Ext']] + parameters[inx[away]['Ext']]
            lik -= bp_logpmf(result[0], result[1], l1, l2, l3)

    return lik

def simulate_shock_model(lambda_1, lambda_2, theta, n_sims):
    U = np.random.random(n_sims)
    simulations = poisson.rvs(((1 - theta) * lambda_1, (1 - theta) * lambda_2), size = (n_sims, 2))
    simulations[:, 0] += poisson.ppf(U, theta * lambda_1).astype(int)
    simulations[:, 1] += poisson.ppf(U, theta * lambda_2).astype(int)

    return simulations

class ShockModel:
    def __init__(self, competition, year, n_sims = 5_000_000):
        self.competition = competition
        self.n_sims = n_sims
        self.year = year

    def optimize_parameters(self):
        played_games, inx, games = preprocessing(self.competition, self.year, True)
        filename = f'parameters/{self.competition}_{self.year}_shock_model.json'
        if filename in os.listdir(): parameters = generate_x0(filename, inx, True)
        else: parameters = np.random.random(3 * len(played_games))
        bounds = [(0, 0.1) if i % 3 == 2 else (0, None) for i in range(len(parameters))]
        bounds[0] = (1, 1)
        res = minimize(bp_likelihood, parameters, args = (played_games, inx), bounds = bounds)
        parameters = res.x
        for club in inx:
            for force in inx[club]:
                inx[club][force] = parameters[inx[club][force]]

        parameters = inx
        with open(filename, 'w') as f: json.dump(parameters, f)
        games = [game.strip().split(' vs. ') for game in games]
        return parameters, games, played_games

    def simulation(self, games, played_games, parameters):
        points = dict()
        game_probs = dict()
        for i, club in enumerate(played_games):
            points[club] = np.zeros((self.n_sims, 4), dtype = int)
            points[club] = np.hstack([points[club], np.arange(self.n_sims).reshape(-1, 1)])
            points[club] = np.hstack([points[club], i * np.ones(self.n_sims).reshape(-1, 1)])
        
        for game in tqdm(games):
            home, away = game
            if away in played_games[home]:
                home_score, away_score = played_games[home][away]
                points = update_points_table(points, home, away, home_score, away_score)
            else:
                if home not in game_probs: game_probs[home] = dict()
                game_probs[home][away] = np.zeros((6, 6))
                theta = parameters[home]['Ext'] + parameters[away]['Ext']
                lambda_1 = parameters[home]['Atk'] / parameters[away]['Def']
                lambda_2 = parameters[away]['Atk'] / parameters[home]['Def']
                simulations = simulate_shock_model(lambda_1, lambda_2, theta, self.n_sims)

                # points
                points = update_points_table(points, home, away, simulations[:, 0], simulations[:, 1])
                scores, counts = np.unique(simulations, axis = 0, return_counts = True)
                for i in range(len(scores)):
                    if scores[i][0] > 5: scores[i][0] = 5
                    if scores[i][1] > 5: scores[i][1] = 5
                    game_probs[home][away][scores[i][0], scores[i][1]] += counts[i]

                game_probs[home][away] = game_probs[home][away] / self.n_sims
                del simulations

        for home in game_probs:
            for away in game_probs[home]:
                game_probs[home][away] = game_probs[home][away].tolist()
        
        return generate_table(points), game_probs

    def run_model(self):
        parameters, games, played_games = self.optimize_parameters()
        table, game_probs = self.simulation(games, played_games, parameters)
        table = calculate_final_position(table)
        with open(f'results/probs/game_probs_{self.competition}_{self.year}_shock_model.json', 'w') as f: json.dump(game_probs, f)

        probs = (table.groupby(['Club', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Position', ascending = False, ignore_index = True)

        probs['Cumulative'] = probs.groupby('Club').cumsum()['Probability']
        probs.to_csv(f'results/probs/{self.competition}_{self.year}_shock_model.csv', index = False)
        plot_probs(probs, 'Probability by Club and Position (Shock Model)', f'{self.competition}_{self.year}_shock_model')

        stats = (table.groupby(['Points', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Points', ascending = True, ignore_index = True)

        stats['Cumulative'] = stats.groupby('Position').cumsum()['Probability']
        stats.to_csv(f'results/stats/{self.competition}_{self.year}_shock_model.csv', index = False)