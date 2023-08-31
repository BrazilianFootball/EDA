import os
import stan
import numpy as np
import httpstan

from scipy.stats import poisson

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
data {
    int<lower = 1> n_obs;
    int<lower = 1> n_clubs;
    array[n_obs, 2] int results;
    array[n_obs, 2] int clubs;
}

parameters {
    array[n_clubs] real<lower = 0> lambda_atk;
    array[n_clubs] real<lower = 0> lambda_def;
}

model {
    lambda_atk ~ gamma(0.01, 0.01);
    lambda_def ~ gamma(0.01, 0.01);
    for (n in 1:n_obs) {
        results[n, 1] ~ poisson(lambda_atk[clubs[n, 1]] / lambda_def[clubs[n, 2]]);
        results[n, 2] ~ poisson(lambda_atk[clubs[n, 2]] / lambda_def[clubs[n, 1]]);
    }
}
'''

n_obs = 500
lambda_1, lambda_2 = 2, 1
results = poisson.rvs((lambda_1, lambda_2), size = (n_obs, 2))
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

filename = 'results/simple_model_posterior.csv'
df = run(model, data, filename, num_samples = 4000, num_warmup = 1000, reset = True)
print(df.describe())