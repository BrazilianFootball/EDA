import os
import json
import pickle
import warnings
import numpy as np

from tqdm import tqdm
from scipy.stats import poisson
from scipy.optimize import minimize

from utils.functions import *

warnings.filterwarnings('ignore')

def simulate_shock_model(lambda_1, lambda_2, theta, n_sims):
    U = np.random.random(n_sims)
    simulations = poisson.rvs(((1 - theta) * lambda_1, (1 - theta) * lambda_2), size = (n_sims, 2))
    simulations[:, 0] += poisson.ppf(U, theta * lambda_1).astype(int)
    simulations[:, 1] += poisson.ppf(U, theta * lambda_2).astype(int)

    return simulations

class ShockModel:
    def __init__(self, competition, year, n_sims = 5_000_000, max_games = 380, ignored_games = list(), x0 = None, home_away_pars = 0, to_git = True, max_iters = 10, max_theta = .2):
        self.x0 = x0
        self.iters = 1
        self.year = year
        self.to_git = to_git
        self.n_sims = n_sims
        self.max_theta = max_theta
        self.max_iters = max_iters
        self.max_games = max_games
        self.competition = competition
        self.ignored_games = ignored_games
        self.home_away_pars = home_away_pars
        self.filename_tag = f'{self.competition}_{self.year}_shock_model_{self.max_games}_games_{60 + self.home_away_pars}_pars'
        assert self.home_away_pars in [0, 1, 20, 40], 'Number of parameters to home effect must be one of these: 0, 1, 20 or 40.'

    def bp_cdf(self, x1, x2, l1, l2, theta):
        i = np.arange(x1 + 1)
        j = np.arange(x2 + 1)
        m1 = np.minimum.outer(poisson.cdf(i, theta * l1), poisson.cdf(j, theta * l2))
        m2 = np.multiply.outer(poisson.pmf(x1 - i, (1 - theta) * l1), poisson.pmf(x2 - j, (1 - theta) * l2))
        return (m1 * m2).sum()

    def bp_pmf(self, x1, x2, l1, l2, theta):
        return self.bp_cdf(x1, x2, l1, l2, theta) \
                - self.bp_cdf(x1 - 1, x2, l1, l2, theta) \
                - self.bp_cdf(x1, x2 - 1, l1, l2, theta) \
                + self.bp_cdf(x1 - 1, x2 - 1, l1, l2, theta)

    def bp_logpmf(self, x1, x2, l1, l2, theta):
        return np.log(self.bp_pmf(x1, x2, l1, l2, theta))

    def bp_likelihood(self, parameters, played_games, inx):
        lik = 0
        if self.home_away_pars == 0:
            for home in played_games:
                for away in played_games[home]:
                    result = played_games[home][away]
                    inx_1, inx_2 = inx[home]['Atk'] - 1, inx[away]['Def'] - 1
                    if inx_1 == -1: l1 = 1 / parameters[inx_2]
                    else: l1 = parameters[inx_1] / parameters[inx_2]
                    
                    inx_1, inx_2 = inx[away]['Atk'] - 1, inx[home]['Def'] - 1
                    if inx_1 == -1: l2 = 1 / parameters[inx_2]
                    else: l2 = parameters[inx_1] / parameters[inx_2]

                    inx_1, inx_2 = inx[home]['Ext'] - 1, inx[away]['Ext'] - 1
                    l3 = parameters[inx_1] + parameters[inx_2]
                    lik -= self.bp_logpmf(result[0], result[1], l1, l2, l3)

        elif self.home_away_pars == 1:
            for home in played_games:
                for away in played_games[home]:
                    result = played_games[home][away]
                    inx_1, inx_2 = inx[home]['Atk'] - 1, inx[away]['Def'] - 1
                    if inx_1 == -1: l1 = 1 / parameters[inx_2] + parameters[-1]
                    else: l1 = parameters[inx_1] / parameters[inx_2] + parameters[-1]

                    inx_1, inx_2 = inx[away]['Atk'] - 1, inx[home]['Def'] - 1
                    if inx_1 == -1: l2 = 1 / parameters[inx_2]
                    else: l2 = parameters[inx_1] / parameters[inx_2]

                    inx_1, inx_2 = inx[home]['Ext'] - 1, inx[away]['Ext'] - 1
                    l3 = parameters[inx_1] + parameters[inx_2]
                    lik -= self.bp_logpmf(result[0], result[1], l1, l2, l3)

        elif self.home_away_pars == 20:
            for home in played_games:
                for away in played_games[home]:
                    result = played_games[home][away]
                    inx_1, inx_2, inx_3 = inx[home]['Atk'] - 1, inx[away]['Def'] - 1, inx[home]['Home bonus'] - 1
                    if inx_1 == -1: l1 = 1 / parameters[inx_2] + parameters[inx_3]
                    else: l1 = parameters[inx_1] / parameters[inx_2] + parameters[inx_3]
                    
                    inx_1, inx_2 = inx[away]['Atk'] - 1, inx[home]['Def'] - 1
                    if inx_1 == -1: l2 = 1 / parameters[inx_2]
                    else: l2 = parameters[inx_1] / parameters[inx_2]
                    
                    inx_1, inx_2 = inx[home]['Ext'] - 1, inx[away]['Ext'] - 1
                    l3 = parameters[inx_1] + parameters[inx_2]
                    lik -= self.bp_logpmf(result[0], result[1], l1, l2, l3)

        elif self.home_away_pars == 40:
            for home in played_games:
                for away in played_games[home]:
                    result = played_games[home][away]
                    inx_1, inx_2 = inx[home]['Home']['Atk'] - 1, inx[away]['Away']['Def'] - 1
                    if inx_1 == -1: l1 = 1 / parameters[inx_2]
                    else: l1 = parameters[inx_1] / parameters[inx_2]

                    inx_1, inx_2 = inx[away]['Away']['Atk'] - 1, inx[home]['Home']['Def'] - 1
                    l2 = parameters[inx_1] / parameters[inx_2]

                    inx_1, inx_2 = inx[home]['Ext'] - 1, inx[away]['Ext'] - 1
                    l3 = parameters[inx_1] + parameters[inx_2]
                    lik -= self.bp_logpmf(result[0], result[1], l1, l2, l3)

        return lik

    def preprocessing(self):
        with open(f'../data/BrazilianSoccerData/results/processed/{self.competition}_{self.year}_games.json', 'r') as f:
            data = json.load(f)

        inx = dict()
        played_games = dict()
        inx_count = 0
        bounds = list()
        for game in data:
            if int(game) > self.max_games or int(game) in self.ignored_games: continue
            game = str(game).zfill(3)
            home, away, result = data[game]['Home'], data[game]['Away'], data[game]['Result']
            result = result.upper().split(' X ')
            result = [int(x) for x in result]
            if home not in played_games: played_games[home] = dict()
            played_games[home][away] = result
            if home not in inx:
                inx[home] = dict()
                if self.home_away_pars == 40:
                    inx[home]['Home'] = dict()
                    inx[home]['Away'] = dict()
                    inx[home]['Home']['Atk'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                    inx[home]['Home']['Def'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                    inx[home]['Away']['Atk'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                    inx[home]['Away']['Def'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                else:
                    inx[home]['Atk'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                    inx[home]['Def'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1
                
                if self.home_away_pars == 20:
                    inx[home]['Home bonus'] = inx_count
                    bounds.append((0, None))
                    inx_count += 1

                inx[home]['Ext'] = inx_count
                bounds.append((0, self.max_theta / 2))
                inx_count += 1

        # bounds[0] = (1, 1)
        bounds.pop(0)
        if self.home_away_pars == 1: bounds.append((0, None))
        if 'data' not in os.listdir('results'): os.mkdir('results/data')
        games = generate_games(list(played_games.keys()), f'results/data/{self.competition}_{self.year}.csv')
        return played_games, inx, games, bounds

    def optimize_parameters(self, verbose):
        played_games, inx, games, bounds = self.preprocessing()
        filename = f'parameters/{self.filename_tag}.json'
        if self.x0 is not None and self.x0.shape[0] == (3 * len(played_games) + self.home_away_pars - 1): parameters = self.x0
        else: parameters = np.random.random(3 * len(played_games) + self.home_away_pars - 1)
        res = minimize(self.bp_likelihood, parameters, args = (played_games, inx), bounds = bounds)
        with open(f'results/optimizer/optimizer_result_{self.filename_tag}.pkl', 'wb') as f: pickle.dump(res, f)
        if not res.success and verbose:
            print(f'Parameters didn\'t converge ({self.iters}/{self.max_iters})')
        
        parameters = np.hstack([np.array([1]), res.x])
        if self.home_away_pars == 40:
            for club in inx:
                for local in inx[club]:
                    if local == 'Ext':
                        inx[club][local] = parameters[inx[club][local]]
                        continue
                    
                    for force in inx[club][local]:
                        inx[club][local][force] = parameters[inx[club][local][force]]
        else:
            for club in inx:
                for force in inx[club]:
                    inx[club][force] = parameters[inx[club][force]]

        if self.home_away_pars == 1: inx['Home bonus'] = parameters[-1]
        parameters = inx
        with open(filename, 'w') as f: json.dump(parameters, f)
        games = [game.strip().split(' vs. ') for game in games]
        return res.success, parameters, games, played_games

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
                if self.home_away_pars == 40:
                    lambda_1 = parameters[home]['Home']['Atk'] / parameters[away]['Away']['Def']
                    lambda_2 = parameters[away]['Away']['Atk'] / parameters[home]['Home']['Def']
                else:
                    lambda_1 = parameters[home]['Atk'] / parameters[away]['Def']
                    lambda_2 = parameters[away]['Atk'] / parameters[home]['Def']

                if self.home_away_pars == 1: lambda_1 += parameters['Home bonus']
                elif self.home_away_pars == 20: lambda_1 += parameters[home]['Home bonus']
                theta = parameters[home]['Ext'] + parameters[away]['Ext']
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

    def run_model(self, verbose = True, show_fig = True):
        success, parameters, games, played_games = self.optimize_parameters(verbose)
        while not success and self.iters < self.max_iters:
            self.iters += 1
            success, parameters, games, played_games = self.optimize_parameters(verbose)

        if not success: self.n_sims = int(self.n_sims ** (1/2))
        elif verbose: print(f'Parameters converged after {self.iters} iteration(s)!')
        table, game_probs = self.simulation(games, played_games, parameters)
        table = calculate_final_position(table)
        with open(f'results/probs/game_probs_{self.filename_tag}.json', 'w') as f: json.dump(game_probs, f)

        probs = (table.groupby(['Club', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Position', ascending = False, ignore_index = True)

        probs['Cumulative'] = probs.groupby('Club').cumsum()['Probability']
        probs.to_csv(f'results/probs/{self.filename_tag}.csv', index = False)
        title = f'Probability by Club and Position - Shock Model<br><span style="font-size: 14px;">Results of a model with {60 + self.home_away_pars} parameters and {self.n_sims:,} simulations for each game</span>'
        plot_probs(probs, title, self.filename_tag, to_git = self.to_git, show_fig = show_fig)

        stats = (table.groupby(['Points', 'Position']).count()['Simulation'] / self.n_sims) \
            .reset_index() \
            .rename({'Simulation' : 'Probability'}, axis = 1) \
            .sort_values('Points', ascending = True, ignore_index = True)

        stats['Cumulative'] = stats.groupby('Position').cumsum()['Probability']
        stats.to_csv(f'results/stats/{self.filename_tag}.csv', index = False)