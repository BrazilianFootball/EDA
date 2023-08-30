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

def likelihood(parameters, played_games, inx):
    lik = 0
    for home in played_games:
        for away in played_games[home]:
            result = played_games[home][away]
            mu0 = parameters[inx[home]['Atk']] / parameters[inx[away]['Def']]
            mu1 = parameters[inx[away]['Atk']] / parameters[inx[home]['Def']]
            lik -= poisson.logpmf(result[0], mu0)
            lik -= poisson.logpmf(result[1], mu1)

    return lik

class IndependentsPoissonModel:
    def __init__(self, competition, year, n_sims = 5_000_000):
        self.competition = competition
        self.n_sims = n_sims
        self.year = year

    def optimize_parameters(self):
        played_games, inx, games = preprocessing(self.competition, self.year, False)
        filename = f'parameters/{self.competition}_{self.year}.json'
        if filename in os.listdir(): parameters = generate_x0(filename, inx, False)
        else: parameters = np.random.random(2 * len(played_games))
        bounds = [(0, None) for _ in parameters]
        bounds[0] = (1, 1)
        res = minimize(likelihood, parameters, args = (played_games, inx), bounds = bounds)
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
                mu0 = parameters[home]['Atk'] / parameters[away]['Def']
                mu1 = parameters[away]['Atk'] / parameters[home]['Def']
                simulations = poisson.rvs((mu0, mu1), size = (self.n_sims, 2))

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
        with open(f'results/probs/game_probs_{self.competition}_{self.year}.json', 'w') as f: json.dump(game_probs, f)

        probs = (table.groupby(['Club', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Position', ascending = False, ignore_index = True)

        probs['Cumulative'] = probs.groupby('Club').cumsum()['Probability']
        probs.to_csv(f'results/probs/{self.competition}_{self.year}.csv', index = False)
        plot_probs(probs, 'Probability by Club and Position (Independents Poisson)', f'{self.competition}_{self.year}')

        stats = (table.groupby(['Points', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Points', ascending = True, ignore_index = True)

        stats['Cumulative'] = stats.groupby('Position').cumsum()['Probability']
        stats.to_csv(f'results/stats/{self.competition}_{self.year}.csv', index = False)