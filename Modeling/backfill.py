import os
import re
import sys
import pickle
import datetime

from glob import glob
from utils.mail_delivery import send
from utils.functions import create_directories

from utils.shock_model import ShockModel
from utils.holgates_bivariate_poisson import HolgatesPoissonModel
from utils.independents_poisson_model import IndependentsPoissonModel

if __name__ == '__main__':
    create_directories()
    cleaning = True if '--c' in sys.argv else False
    home_away_pars_list = [0, 1, 20, 40]
    today = datetime.date.today()
    competitions = ['Serie_A', 'Serie_B']
    n_sims = 1_000_000
    year = today.year
    max_games = 380

    for arg in sys.argv:
        if '-g=' in arg: max_games = int(arg.split('=')[-1])
        if '-n=' in arg: n_sims = int(arg.split('=')[-1])
        if '-y=' in arg: year = int(arg.split('=')[-1])

        if '-m=' in arg: from_mail = arg.split('=')[-1]
        if '-p=' in arg: password = arg.split('=')[-1]

    subject = '[BSM] Backfill'
    body = ''

    models = {'Independents Poisson' : IndependentsPoissonModel,
              'Holgates Bivariate Poisson' : HolgatesPoissonModel,
              'Shock Model' : ShockModel}
    
    attachments = list()
    files_to_remove = list()
    for competition in competitions:
        if competition == 'Serie_A' and year == 2023 and max_games > 220: continue
        elif competition == 'Serie_B' and year == 2023 and max_games > 270: continue
        for model in models:
            for home_away_pars in home_away_pars_list:
                if cleaning: os.system('clear')
                print(f"{model} - {competition.replace('_', ' ')} {year} - {home_away_pars} home/away parameters")
                cur_model = models[model](competition, year, n_sims, home_away_pars = home_away_pars, max_games = max_games, max_iters = 100)
                if f'optimizer_result_{cur_model.filename_tag}.pkl' in os.listdir('results/optimizer/'):
                    with open(f'results/optimizer/optimizer_result_{cur_model.filename_tag}.pkl', 'rb') as f:
                        res = pickle.load(f)
                    
                    pars = int(re.findall('_(\d+)_pars', cur_model.filename_tag)[0])
                    if (res.x.shape[0] == (pars - 1)) and (res.success) and (res.hess_inv is not None):
                        files_to_remove += glob(f'results/*/*{cur_model.filename_tag}*')
                        files_to_remove += glob(f'parameters/{cur_model.filename_tag}.json')
                        for file in files_to_remove: os.remove(file)
                        files_to_remove = list()
                        continue

                cur_model.run_model(show_fig = False)
                attachments += glob(f'results/*/*{cur_model.filename_tag}*')
                attachments += glob(f'parameters/{cur_model.filename_tag}.json')
                if len(attachments) != 0 and model == 'Shock Model':
                    send(from_mail, password, from_mail, subject, body, attachments)
                    for file in attachments: os.remove(file)
                    attachments = list()
                
            if len(attachments) != 0:
                send(from_mail, password, from_mail, subject, body, attachments)
                for file in attachments: os.remove(file)
                attachments = list()