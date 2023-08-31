import os
import stan
import numpy as np
import httpstan

from utils.shock_model import simulate_shock_model

os.system('clear')
def clean_cache(model):
    model_name = httpstan.models.calculate_model_name(model)
    httpstan.cache.delete_model_directory(model_name)

def run(model, data, filename, num_samples = 1000, num_warmup = 1000, reset = True):
    if reset: clean_cache(model)
    posterior = stan.build(model, data = data, random_seed = 0)
    fit = posterior.sample(num_chains = 1, num_samples = num_samples, num_warmup = num_warmup)
    df = fit.to_frame()
    df.to_csv(filename)
    return df

model = '''
functions {
    real bp_cdf(int x1, int x2, real lambda_1, real lambda_2, real theta) {
        real a;
        real b;
        matrix[x1 + 1, x2 + 1] m1;
        matrix[x1 + 1, x2 + 1] m2;

        for (i in 0:x1) {
            for (j in 0:x2) {
                a = poisson_cdf(i | theta * lambda_1);
                b = poisson_cdf(j | theta * lambda_2);
                if (a < b) {
                    m1[i + 1, j + 1] = a;
                } else {
                    m1[i + 1, j + 1] = b;
                }
                // m1[i + 1, j + 1] = min(poisson_cdf(i | theta * lambda_1), poisson_cdf(j | theta * lambda_2));
                m2[i + 1, j + 1] = (poisson_cdf(x1 - i | (1 - theta) * lambda_1) - poisson_cdf(x1 - i - 1 | (1 - theta) * lambda_1)) * (poisson_cdf(x2 - j | (1 - theta) * lambda_2) - poisson_cdf(x2 - j - 1 | (1 - theta) * lambda_2));
            }
        }

        real result = sum(m1 .* m2);
        return result;
    }

    real bp_lpmf(array[] int x, real lambda_1, real lambda_2, real theta) {
        return bp_cdf(x[1], x[2], lambda_1, lambda_2, theta) -
               bp_cdf(x[1] - 1, x[2], lambda_1, lambda_2, theta) -
               bp_cdf(x[1], x[2] - 1, lambda_1, lambda_2, theta) +
               bp_cdf(x[1] - 1, x[2] - 1, lambda_1, lambda_2, theta);
    }
}

data {
    int<lower = 1> n_obs;
    int<lower = 1> n_clubs;
    array[n_obs, 2] int results;
    array[n_obs, 2] int clubs;
}

parameters {
    array[n_clubs] real<lower = 0> lambda_atk;
    array[n_clubs] real<lower = 0> lambda_def;
    array[n_clubs] real<lower = 0, upper = 0.5> theta;
}

model {
    lambda_atk ~ gamma(0.01, 0.01);
    lambda_def ~ gamma(0.01, 0.01);
    theta ~ gamma(0.01, 0.01);
    for (n in 1:n_obs) {
        target += bp_lpmf(results[n] |
                          lambda_atk[clubs[n, 1]] / lambda_def[clubs[n, 2]],
                          lambda_atk[clubs[n, 2]] / lambda_def[clubs[n, 1]],
                          theta[clubs[n, 1]] + theta[clubs[n, 2]]);
    }
}
'''

n_obs = 50
lambda_1, lambda_2, theta = 2, 1, .2
results = simulate_shock_model(lambda_1, lambda_2, theta, n_obs)
games = [[1, 2], [2, 1],
         [1, 3], [3, 1],
         [1, 4], [4, 1],
         [2, 3], [3, 2],
         [2, 4], [4, 2],
         [3, 4], [4, 3]]
clubs = np.array([games[np.random.randint(len(games))] for _ in range(n_obs)])
n_clubs = np.max(clubs)
data = {'n_obs': n_obs,
        'n_clubs': n_clubs,
        'results': results,
        'clubs': clubs}

filename = 'results/shock_model_posterior.csv'
df = run(model, data, filename, num_samples = 4000, num_warmup = 1000, reset = True)
print(df.describe())