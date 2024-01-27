data {
    int<lower=1> num_jogos;  // Número total de jogos
    int<lower=1> num_equipes;  // Número total de equipes
    array[num_jogos] int<lower=1, upper=num_equipes> equipe1;  // Equipe 1 em cada jogo
    array[num_jogos] int<lower=1, upper=num_equipes> equipe2;  // Equipe 2 em cada jogo
    array[num_jogos] int<lower=0> gols_equipe1;  // Gols da equipe 1 em cada jogo
    array[num_jogos] int<lower=0> gols_equipe2;  // Gols da equipe 2 em cada jogo
}

parameters {
    vector<lower=0>[num_equipes] habilidade;  // Habilidade de cada equipe
}

model {
    habilidade ~ normal(0, 1);  // Prior normal para as habilidades
    for (jogo in 1:num_jogos) {
        // Likelihood
        target += gols_equipe1[jogo] * (log(habilidade[equipe1[jogo]]) - log(habilidade[equipe2[jogo]])) - (habilidade[equipe1[jogo]] / habilidade[equipe2[jogo]]);
        target += gols_equipe2[jogo] * (log(habilidade[equipe2[jogo]]) - log(habilidade[equipe1[jogo]])) - (habilidade[equipe2[jogo]] / habilidade[equipe1[jogo]]);
    }
}