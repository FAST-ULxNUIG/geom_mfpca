# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

library(MFPCA)
library(tidyverse)

# ------------------------------------------------------------------------------
# Parameters
N_sim <- 50
N <- 100
M <- 20
npc <- 0.95

# ------------------------------------------------------------------------------
# Exponential
evals_list_split <- vector("list", length = N_sim)
evals_list_weighted <- vector("list", length = N_sim)

true_vals <- eVal(M, 'exponential')
true_vals_df <- tibble(
    variable = 1:length(true_vals),
    value = true_vals
)

for (i in seq_len(N_sim)) {
    print(paste("Iteration", i))
    sim <-  simMultiFunData(
        type = "split",
        argvals = list(seq(0,1,0.01), seq(0,1,0.01)),
        M = M,
        eFunType = "Fourier",
        eValType = "exponential",
        N = 100)
    pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = npc)
    pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = npc)
    MFPCA_est <- MFPCA(
        sim$simData,
        M = pace1$npc + pace2$npc, 
        uniExpansions = list(
            list(type = "uFPCA", pve = npc),
            list(type = "uFPCA", pve = npc)),
        fit = TRUE)
    evals_list_split[[i]] <- MFPCA_est$values
    
    sim <-  simMultiFunData(
        type = "weighted",
        argvals = list(list(seq(0, 1, 0.01)), list(seq(0, 1, 0.01))),
        M = list(M, M),
        eFunType = list("Fourier", "Wiener"),
        eValType = "exponential",
        N = N)
    pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = npc)
    pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = npc)
    MFPCA_est <- MFPCA(
        sim$simData,
        M = pace1$npc + pace2$npc, 
        uniExpansions = list(
            list(type = "uFPCA", pve = npc),
            list(type = "uFPCA", pve = npc)),
        fit = TRUE)
    evals_list_weighted[[i]] <- MFPCA_est$values
}


split_df <- plyr::ldply(evals_list_split, rbind) |> 
    reshape2::melt()
ggplot(split_df) +
    geom_boxplot(aes(x = variable, y = value)) +
    geom_point(aes(x = variable, y = value), data = true_vals_df, col = 'red') +
    xlab('Eigenvalue number') +
    ylab('Eigenvalue') +
    theme_minimal()

weighted_df <- plyr::ldply(evals_list_weighted, rbind) |> 
    reshape2::melt()
ggplot(weighted_df) +
    geom_boxplot(aes(x = variable, y = value)) +
    geom_point(aes(x = variable, y = value), data = true_vals_df, col = 'red') +
    xlab('Eigenvalue number') +
    ylab('Eigenvalue') +
    theme_minimal()

# PVE
true_vals_df |> 
    summarise(pve = cumsum(value) / sum(true_vals))
split_df |> 
    group_by(variable) |>
    summarise(m = mean(value)) |>
    summarise(pve = cumsum(m) / sum(true_vals))
weighted_df |> 
    group_by(variable) |>
    summarise(m = mean(value)) |>
    summarise(pve = cumsum(m) / sum(true_vals))

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Linear
evals_list_split <- vector("list", length = N_sim)
evals_list_weighted <- vector("list", length = N_sim)

true_vals <- eVal(M, 'linear')
true_vals_df <- tibble(
    variable = 1:length(true_vals),
    value = true_vals
)

for (i in seq_len(N_sim)) {
    print(paste("Iteration", i))
    sim <-  simMultiFunData(
        type = "split",
        argvals = list(seq(0,1,0.01), seq(0,1,0.01)),
        M = M,
        eFunType = "Fourier",
        eValType = "linear",
        N = 100)
    pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = npc)
    pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = npc)
    MFPCA_est <- MFPCA(
        sim$simData,
        M = pace1$npc + pace2$npc, 
        uniExpansions = list(
            list(type = "uFPCA", pve = npc),
            list(type = "uFPCA", pve = npc)),
        fit = TRUE)
    evals_list_split[[i]] <- MFPCA_est$values
    
    sim <-  simMultiFunData(
        type = "weighted",
        argvals = list(list(seq(0, 1, 0.01)), list(seq(0, 1, 0.01))),
        M = list(M, M),
        eFunType = list("Fourier", "Wiener"),
        eValType = "linear",
        N = N)
    pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = npc)
    pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = npc)
    MFPCA_est <- MFPCA(
        sim$simData,
        M = pace1$npc + pace2$npc, 
        uniExpansions = list(
            list(type = "uFPCA", pve = npc),
            list(type = "uFPCA", pve = npc)),
        fit = TRUE)
    evals_list_weighted[[i]] <- MFPCA_est$values
}


split_df <- plyr::ldply(evals_list_split, rbind) |> 
    reshape2::melt()
ggplot(split_df) +
    geom_boxplot(aes(x = variable, y = value)) +
    geom_point(aes(x = variable, y = value), data = true_vals_df, col = 'red') +
    xlab('Eigenvalue number') +
    ylab('Eigenvalue') +
    theme_minimal()

weighted_df <- plyr::ldply(evals_list_weighted, rbind) |> 
    reshape2::melt()
ggplot(weighted_df) +
    geom_boxplot(aes(x = variable, y = value)) +
    geom_point(aes(x = variable, y = value), data = true_vals_df, col = 'red') +
    xlab('Eigenvalue number') +
    ylab('Eigenvalue') +
    theme_minimal()

# PVE
true_vals_df |> 
    summarise(pve = cumsum(value) / sum(true_vals))
split_df |> 
    group_by(variable) |>
    summarise(m = mean(value)) |>
    summarise(pve = cumsum(m) / sum(true_vals))
weighted_df |> 
    group_by(variable) |>
    summarise(m = mean(value)) |>
    summarise(pve = cumsum(m) / sum(true_vals))
