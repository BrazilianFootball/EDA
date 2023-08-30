
import datetime
from utils.shock_model import ShockModel
from utils.holgates_bivariate_poisson import HolgatesPoissonModel
from utils.independents_poisson_model import IndependentsPoissonModel

if __name__ == '__main__':
    today = datetime.date.today()
    year = today.year

    ShockModel('Serie_A', year, 1_000_000).run_model()
    ShockModel('Serie_B', year, 1_000_000).run_model()

    HolgatesPoissonModel('Serie_A', year, 1_000_000).run_model()
    HolgatesPoissonModel('Serie_B', year, 1_000_000).run_model()

    IndependentsPoissonModel('Serie_A', year, 1_000_000).run_model()
    IndependentsPoissonModel('Serie_B', year, 1_000_000).run_model()