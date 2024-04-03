import json
import networkx as nx
import matplotlib.pyplot as plt

from glob import glob
from itertools import product

players_connectivity = dict()
for file in sorted(glob('../../Data/results/processed/Serie_A_*_squads.json')):
    year = int(file[-16:-12])
    if year == 2024: continue
    with open(file, 'rb') as f: data = json.load(f)
    for game in data:
        for sub_game in data[game]:
            home_squad = data[game][sub_game]['Home']['Squad']
            away_squad = data[game][sub_game]['Away']['Squad']
            total_time = data[game][sub_game]['Time']
            for player_A, player_B in product(home_squad, repeat=2):
                if player_A not in players_connectivity: players_connectivity[player_A] = dict()
                if player_B not in players_connectivity: players_connectivity[player_B] = dict()
                
                if player_B in players_connectivity[player_A]: players_connectivity[player_A][player_B] += total_time
                else: players_connectivity[player_A][player_B] = total_time

                if player_A in players_connectivity[player_B]: players_connectivity[player_B][player_A] += total_time
                else: players_connectivity[player_B][player_A] = total_time

fig, ax = plt.subplots(figsize = (100, 100))

G = nx.Graph()
for player in players_connectivity:
    G.add_node(player, color='steelblue', weight=players_connectivity[player][player])

for player_A, player_B in product(players_connectivity, repeat=2):
    if player_A == player_B: continue
    if player_B not in players_connectivity[player_A]: continue
    G.add_edge(player_A, player_B, color='lightskyblue',
               width=players_connectivity[player_A][player_B],
               weight=players_connectivity[player_A][player_B])

print('graph already created')
pos = nx.circular_layout(G)

print(f'drawing graph with {len(players_connectivity)} nodes')
nx.draw_networkx(G,
                 ax=ax,
                 pos=pos,
                 with_labels=False,
                 node_color='steelblue',
                 edge_color='lightskyblue',
                 node_size=[nx.get_node_attributes(G, 'weight')[g] * 0.25 for g in G.nodes()],
                 width=[nx.get_edge_attributes(G, 'width')[g] * 0.001 for g in G.edges()])

plt.savefig(f'../figures/players_connectivity.png')

print('finding communities')
communities = nx.community.greedy_modularity_communities(G, weight='weight')
n_communities = len(communities)
print(f'We found a total of {n_communities} community(ies)')
