import os
import warnings
import pandas as pd
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

def create_directories():
    try: os.chdir('results')
    except:
        os.mkdir('results')
        os.chdir('results')

    for dir in ['images', 'optimizer', 'probs', 'stats']:
        try: os.chdir(dir)
        except:
            os.mkdir(dir)
            os.chdir(dir)
        finally: os.chdir('..')

    os.chdir('..')
    
    try: os.chdir('parameters')
    except:
        os.mkdir('parameters')
        os.chdir('parameters')
    finally: os.chdir('..')

def generate_games(clubs, filename):
    games = list()
    for club1 in clubs:
        for club2 in clubs:
            if club1 == club2: continue
            games.append(f'{club1} vs. {club2}\n')

    with open(filename, 'w') as f: f.writelines(games)
    return games

def generate_table(points):
    cols = ['Points', 'Wins', 'Goals', 'Goals difference', 'Simulation', 'Club_id']
    clubs = {'Club_id' : list(), 'Club' : list()}
    for i, club in enumerate(points):
        clubs['Club_id'].append(i)
        clubs['Club'].append(club)
        if i == 0: table = pd.DataFrame(points[club], columns = cols)
        else: table = pd.concat([table, pd.DataFrame(points[club], columns = cols)])

    del points
    clubs = pd.DataFrame(clubs)
    table = table.astype(int) \
        .merge(clubs, on = 'Club_id') \
        .drop('Club_id', axis = 1)

    return table

def update_points_table(points, home, away, home_score, away_score):
    # points
    points[home][:, 0] += 3 * (home_score > away_score) + 1 * (home_score == away_score)
    points[away][:, 0] += 3 * (away_score > home_score) + 1 * (home_score == away_score)

    # wins
    points[home][:, 1] += 1 * (home_score > away_score)
    points[away][:, 1] += 1 * (away_score > home_score)

    # goals
    points[home][:, 2] += home_score
    points[away][:, 2] += away_score

    # goal difference
    points[home][:, 3] += home_score - away_score
    points[away][:, 3] += away_score - home_score

    return points

def calculate_final_position(table):
    table = table.sort_values(['Points', 'Wins', 'Goals difference', 'Goals'],
                              ascending = [False, False, False, False],
                              ignore_index = True)

    table['Position'] = table.groupby(['Simulation']).cumcount() + 1
    return table

def plot_probs(probs, title, filename, to_git = True, show_fig = True):
    clubs_order = probs \
        .groupby('Club') \
        .max()['Probability'] \
        .reset_index() \
        .merge(probs, on = ['Club', 'Probability']) \
        .sort_values(['Position', 'Cumulative'], ascending = [False, False])['Club'].values

    probs_sorted = pd.DataFrame()
    for club in clubs_order:
        probs_sorted = pd.concat([probs_sorted, probs[probs['Club'] == club]])

    data = go.Heatmap(z = probs_sorted['Probability'],
                      x = probs_sorted['Position'],
                      y = probs_sorted['Club'],
                      colorscale = 'Viridis',
                      colorbar = dict(title = 'Probability'))

    layout = go.Layout(title = title,
                       xaxis = dict(title = 'Position'),
                       yaxis = dict(title = 'Club'),
                       autosize = False,
                       width = 800,
                       height = 600,
                       margin = dict(l = 200, r = 100, b = 50, t = 50))

    fig = go.Figure(data, layout = layout)
    fig.write_image(f'results/images/{filename}.png', format='png')
    if show_fig:
        if to_git: fig.show('png')
        else: fig.show()