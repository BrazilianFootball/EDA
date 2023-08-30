import os
import sys
import datetime
from utils.shock_model import ShockModel
from utils.holgates_bivariate_poisson import HolgatesPoissonModel
from utils.independents_poisson_model import IndependentsPoissonModel

if __name__ == '__main__':
    cleaning = True if '--c' in sys.argv else False
    today = datetime.date.today()
    year = today.year

    if cleaning: os.system('clear')
    print('Independents Poisson - Serie A')
    IndependentsPoissonModel('Serie_A', year, 1_000_000).run_model()
    if cleaning: os.system('clear')
    print('Independents Poisson - Serie B')
    IndependentsPoissonModel('Serie_B', year, 1_000_000).run_model()

    if cleaning: os.system('clear')
    print('Holgates Bivariate Poisson - Serie A')
    HolgatesPoissonModel('Serie_A', year, 1_000_000).run_model()
    if cleaning: os.system('clear')
    print('Holgates Bivariate Poisson - Serie B')
    HolgatesPoissonModel('Serie_B', year, 1_000_000).run_model()

    if cleaning: os.system('clear')
    print('Shock Models - Serie A')
    ShockModel('Serie_A', year, 1_000_000).run_model()
    if cleaning: os.system('clear')
    print('Shock Models - Serie B')
    ShockModel('Serie_B', year, 1_000_000).run_model()