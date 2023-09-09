import os
import sys
from glob import glob
from utils.mail_delivery import catch_results

if __name__ == '__main__':
    cleaning = False
    for arg in sys.argv:
        if '-p=' in arg: password = arg.split('=')[-1]
        if '-m=' in arg: mail = arg.split('=')[-1]
        if '-t=' in arg: tag = arg.split('=')[-1]
        if '--c' in arg: cleaning = True
    
    catch_results(mail, password, tag)
    if cleaning: os.system('clear')
    competitions = ['Serie_A', 'Serie_B']
    years = [*range(2013, 2024)]
    n_games = [*range(100, 390, 10)]
    total_ = 0
    expected_ = 0
    for year in years:
        for competition in competitions:
            total = 0
            expected = 0
            for games in n_games:
                expected += 12
                n = len(glob(f'results/images/{competition}_{year}*{games}_games*'))
                total += n

            total_ += total
            expected_ += expected
            if total != 0: print(f'{year}\'s {competition.replace("_", " ")} is {total / expected:.2%} completed')

    print(f'\nAll is {total_ / expected_:.2%} completed')