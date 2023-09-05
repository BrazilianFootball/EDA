import os
import sys
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
    
    files_to_remove = list()
    for competition in competitions:
        for model in models:
            for home_away_pars in home_away_pars_list:
                if cleaning: os.system('clear')
                print(f"{model} - {competition.replace('_', ' ')} {year} - {home_away_pars} home/away parameters")
                cur_model = models[model](competition, year, n_sims, home_away_pars = home_away_pars, max_games = max_games)
                if cur_model.filename_tag + '.png' in os.listdir('results/images/'):
                    files_to_remove += glob(f'results/*/*{competition}_{year}_*_{max_games}_games_*')
                    files_to_remove += glob(f'parameters/*{competition}_{year}_*_{max_games}_games_*')
                    continue
                else:
                    for file in files_to_remove: os.remove(file)
                    files_to_remove = list()

                cur_model.run_model(show_fig = False)
        
            attachments = glob(f'results/*/*{competition}_{year}_*_{max_games}_games_*')
            attachments += glob(f'parameters/*{competition}_{year}_*_{max_games}_games_*')
            send(from_mail, password, from_mail, subject, body, attachments)
            for file in attachments: os.remove(file)