import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from glob import glob
from copy import deepcopy

import sys
sys.path.append('../../Data/auxiliary/')
from img_path import img_files

def replace_special_char(string):
    changes = [('á', 'a'), ('à', 'a'), ('â', 'a'), ('ã', 'a'),
               ('é', 'e'), ('ê', 'e'), ('í', 'i'), ('ó', 'o'),
               ('ô', 'o'), ('õ', 'o'), ('ú', 'u'), ('ç', 'c')]
    
    for change in changes: string = string.replace(change[0], change[1])
    return string

if __name__ == '__main__':
    club_list = list()
    for file in sorted(glob('../../Data/results/processed/Serie_A_*_games.json')):
        with open(file, 'rb') as f: data = json.load(f)
        for game in data:
            home_club = data[game]['Home']
            away_club = data[game]['Away']
            if home_club not in club_list: club_list.append(home_club)
            if away_club not in club_list: club_list.append(away_club)

    club_list = sorted(club_list, key=lambda x : (x[-2:], x[:-4]))

    clubs_map = dict()
    for club1 in club_list:
        clubs_map[club1] = dict()
        for club2 in club_list:
            clubs_map[club1][club2] = set()

    data_tables = dict()
    for file_index, file in enumerate(sorted(glob('../../Data/results/processed/Serie_A_*_squads.json'))):
        year = int(file[-16:-12])
        if year == 2024: continue
        if file_index == 0: data_tables[year] = clubs_map
        else: data_tables[year] = deepcopy(data_tables[year - 1])

        info_file = file.replace('squads', 'games')
        with open(file, 'rb') as f: data = json.load(f)
        with open(info_file, 'rb') as f: info = json.load(f)
        for game in data:
            home_club = info[game]['Home']
            away_club = info[game]['Away']
            for sub_game in data[game]:
                home_squad = set(data[game][sub_game]['Home']['Squad'])
                away_squad = set(data[game][sub_game]['Away']['Squad'])

                data_tables[year][home_club][home_club] = data_tables[year][home_club][home_club].union(home_squad)
                data_tables[year][away_club][away_club] = data_tables[year][away_club][away_club].union(away_squad)

                for player in home_squad:
                    for club in data_tables[year]:
                        if home_club == club: continue
                        if player in data_tables[year][club][club]:
                            data_tables[year][club][home_club].add(player)

                for player in away_squad:
                    for club in data_tables[year]:
                        if away_club == club: continue
                        if player in data_tables[year][club][club]:
                            data_tables[year][club][away_club].add(player)

    for year in data_tables:
        for club1 in data_tables[year]:
            for club2 in data_tables[year][club1]:
                data_tables[year][club1][club2] = len(data_tables[year][club1][club2])

    clubs_inx = dict()
    for i, club in enumerate(club_list): clubs_inx[club] = i

    table = np.zeros((len(data_tables[year]), len(data_tables[year])), dtype=int)
    for year_index, year in enumerate(data_tables):
        for club1 in data_tables[year]:
            i = clubs_inx[club1]
            for club2 in data_tables[year]:
                j = clubs_inx[club2]
                table[i, j] = data_tables[year][club1][club2]

        w_max = 0
        w_min = 1000
        G = nx.DiGraph()
        for club in clubs_inx:
            i = clubs_inx[club]
            G.add_node(i, color='steelblue', weight=table[i][i])

        for club1 in clubs_inx:
            i = clubs_inx[club1]
            for club2 in clubs_inx:
                if club1 == club2: continue
                j = clubs_inx[club2]
                G.add_edge(i, j, color='lightskyblue', width=table[i][j], weight=table[i][j])

        inx_clubs = {value: key for key, value in clubs_inx.items()}

        fig, ax = plt.subplots(figsize = (50, 50))
        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G,
                                    pos,
                                    edge_labels=nx.get_edge_attributes(G, 'relation'),
                                    label_pos=1.5,
                                    font_size=9,
                                    font_color='red',
                                    font_family='sans-serif',
                                    font_weight='normal',
                                    alpha=1.0,
                                    bbox=None,
                                    ax=ax,
                                    rotate=True)

        nx.draw_networkx(G,
                        pos=pos,
                        ax=ax,
                        node_color=[nx.get_node_attributes(G, 'color')[g] for g in G.nodes()],
                        edge_color=[nx.get_edge_attributes(G, 'color')[g] for g in G.edges()],
                        node_size=[nx.get_node_attributes(G, 'weight')[g] * 10 for g in G.nodes()],
                        width=[nx.get_edge_attributes(G, 'width')[g] * 0.5 for g in G.edges()])

        trans = ax.transData.transform
        trans2 = fig.transFigure.inverted().transform
        weights = nx.get_node_attributes(G, 'weight')
        for club in weights:
            if weights[club] > w_max: w_max = weights[club]
            if weights[club] < w_min: w_min = weights[club]

        dif = w_max - w_min
        new = 0
        relabel = {}
        labels = {}
        for g in G.nodes(): labels[g] = g
        nx.draw_networkx_labels(G,
                                pos=pos,
                                labels=labels,
                                ax=ax,
                                font_color='steelblue')

        for g in G.nodes():
            node = replace_special_char(inx_clubs[g]).replace(' ', '')
            img = mpimg.imread('../../' + img_files[node])
            weight = nx.get_node_attributes(G, 'weight')[g]
            imsize = (weight - w_min) / dif * 0.04 + 0.02
            (x, y) = pos[g]
            xx, yy = trans((x, y))
            xa, ya = trans2((xx, yy))
            a = plt.axes([xa - imsize / 2, ya - imsize / 2, imsize, imsize])
            a.imshow(img)
            a.set_aspect('equal')
            a.axis('off')

        plt.savefig(f'../figures/clubs_connectivity_{year}.png')
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        n_communities = len(communities)
        print(f'We found a total of {n_communities} community(ies) in {year}')
        if n_communities > 1:
            for community_index, community in enumerate(communities):
                print(f'{community_index + 1}° community:')
                for club in community: print(f'  {club_list[club]}')
                print()
