data {
    int<lower=1> n_games;
    int<lower=1> n_teams;
    int<lower=1> n_players_per_team;
    array[n_games] int<lower=0, upper=n_teams> home_team;
    array[n_games] int<lower=0, upper=n_teams> away_team;
    array[n_games] int<lower=0> home_score;
    array[n_games] int<lower=0> away_score;
    array[n_games, 11] int<lower=1> home_players;
    array[n_games, 11] int<lower=1> away_players;
}

parameters {
    vector<lower=0>[n_teams * n_players_per_team] skills;
}

model {
    skills ~ normal(0, 1);
    for (game in 1:n_games) {
        real home_skill = 0;
        real away_skill = 0;
        for (player in 1:11) {
            home_skill += skills[home_team[game] * n_players_per_team + home_players[game, player]];
            away_skill += skills[away_team[game] * n_players_per_team + away_players[game, player]];
        }

        // Likelihood
        target += home_score[game] * (log(home_skill) - log(away_skill)) - (home_skill / away_skill);
        target += away_score[game] * (log(away_skill) - log(home_skill)) - (away_skill / home_skill);
    }
}