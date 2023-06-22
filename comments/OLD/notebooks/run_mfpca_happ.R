
library(funData)
library(MFPCA)


oldPar <- par(no.readonly = TRUE)

N <- 100
K <- 10
npc <- 5

### UFPCA 1D ----
set.seed(42)
sim <- simFunData(
    argvals = seq(0, 1, 0.01),
    M = K,
    eFunType = "Fourier",
    eValType = "exponential",
    N = N
)

fpca <- MFPCA::PACE(sim$simData, npc = npc)

# Save
write.csv(sim$simData@X, './data.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns@X, './true_eigenfunctions.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$mu@X, './estim_mu.csv', row.names = FALSE)
write.csv(fpca$functions@X, './estim_eigenfunctions.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
write.csv(fpca$fit@X, './estim_reconst.csv', row.names = FALSE)
# ----

### UFPCA 2D ----
set.seed(42)
sim <- simFunData(
    argvals = list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
    M = c(K, K),
    eFunType = c("Fourier", "Fourier"),
    eValType = "exponential",
    N = N
)

# without normalization
fpca_nomorm <- MFPCA:::fcptpaBasis(
    funDataObject = sim$simData,
    npc = npc,
    alphaRange = list(v = c(1e-4, 1e4), w = c(1e-4, 1e4)),
    normalize = FALSE
)

# reconstruction
reconst_fpca_nonorm <- array(0, dim = c(N, 101, 101))
for (i in 1:N) {
    reconst_fpca_nonorm[i,,] <- einsum::einsum(
        'i, ijk -> jk', fpca_nomorm$scores[i, ], fpca_nomorm$functions@X
    )
}

sweep(fpca_nomorm$functions@X, MARGIN = 1, fpca_nomorm$scores[i, ], "*")

# with normalization
fpca_morm <- MFPCA:::fcptpaBasis(
    funDataObject = sim$simData,
    npc = npc,
    alphaRange = list(v = c(1e-4, 1e4), w = c(1e-4, 1e4)),
    normalize = TRUE
)

# reconstruction
reconst_fpca_norm <- array(0, dim = c(N, 101, 101))
for (i in 1:N) {
    reconst_fpca_norm[i,,] <- einsum::einsum(
        'i, ijk -> jk', fpca_morm$scores[i, ], fpca_morm$functions@X
    )
}

# Save
write.csv(sim$simData@X, './data.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns@X, './true_eigenfunctions.csv', row.names = FALSE)

write.csv(t(fpca_nomorm$values),
          './estim_eigenvalues_nonorm.csv', row.names = FALSE)
write.csv(fpca_nomorm$functions@X,
          './estim_eigenfunctions_nonorm.csv', row.names = FALSE)
write.csv(fpca_nomorm$scores,
          './estim_scores_nonorm.csv', row.names = FALSE)

write.csv(t(fpca_morm$values),
          './estim_eigenvalues_norm.csv', row.names = FALSE)
write.csv(fpca_morm$functions@X,
          './estim_eigenfunctions_norm.csv', row.names = FALSE)
write.csv(fpca_morm$scores,
          './estim_scores_norm.csv', row.names = FALSE)
# ----

### MFPCA 1D + 1D ----
set.seed(42)
sim <-  simMultiFunData(
    type = "split",
    argvals = list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
    M = K,
    eFunType = "Fourier",
    eValType = "exponential",
    N = N
)

fpca <- MFPCA(
    sim$simData,
    M = npc,
    uniExpansions = list(
        list(type = "uFPCA", npc = K),
        list(type = "uFPCA", npc = K)
    ),
    fit = TRUE
)

# Save
write.csv(sim$simData[[1]]@X, './data_1.csv', row.names = FALSE)
write.csv(sim$simData[[2]]@X, './data_2.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns[[1]]@X, './true_eigenfunctions_1.csv', row.names = FALSE)
write.csv(sim$trueFuns[[2]]@X, './true_eigenfunctions_2.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[1]]@X, './estim_mu_1.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[2]]@X, './estim_mu_2.csv', row.names = FALSE)
write.csv(fpca$functions[[1]]@X,
          './estim_eigenfunctions_1.csv', row.names = FALSE)
write.csv(fpca$functions[[2]]@X,
          './estim_eigenfunctions_2.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
write.csv(fpca$fit[[1]]@X, './estim_reconst_1.csv', row.names = FALSE)
write.csv(fpca$fit[[2]]@X, './estim_reconst_2.csv', row.names = FALSE)
# ----

# MFPCA 2D + 1D ----
set.seed(42)
sim <-  simMultiFunData(
    type = "weighted",
    argvals = list(
        list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
        list(seq(0, 1, 0.01))
    ),
    M = list(c(K/2, K/2), 25),
    eFunType = list(c("Fourier", "Fourier"), "Fourier"),
    eValType = "exponential",
    N = N
)

fpca <- MFPCA(
    sim$simData,
    M = npc,
    uniExpansions = list(
        list(
            type = "FCP_TPA", npc = npc,
            alphaRange = list(v = c(1e-4, 1e4), w = c(1e-4, 1e4))
        ),
        list(type = "uFPCA", npc = npc)
    ), fit = TRUE
)

# Save
write.csv(sim$simData[[1]]@X, './data_1.csv', row.names = FALSE)
write.csv(sim$simData[[2]]@X, './data_2.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns[[1]]@X, './true_eigenfunctions_1.csv', row.names = FALSE)
write.csv(sim$trueFuns[[2]]@X, './true_eigenfunctions_2.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[1]]@X, './estim_mu_1.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[2]]@X, './estim_mu_2.csv', row.names = FALSE)
write.csv(fpca$functions[[1]]@X,
          './estim_eigenfunctions_1.csv', row.names = FALSE)
write.csv(fpca$functions[[2]]@X,
          './estim_eigenfunctions_2.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
# ----


# MFPCA 2D + 2D ----
set.seed(42)
sim <-  simMultiFunData(
    type = "weighted",
    argvals = list(
        list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
        list(seq(0, 1, 0.01), seq(0, 1, 0.01))
    ),
    M = list(c(K/2, K/2), c(K/2, K/2)),
    eFunType = list(c("Fourier", "Fourier"), c("Poly", "Poly")),
    eValType = "exponential",
    N = N
)

fpca <- MFPCA(
    sim$simData,
    M = npc,
    uniExpansions = list(
        list(
            type = "FCP_TPA", npc = npc,
            alphaRange = list(v = c(1e-4, 1e4), w = c(1e-4, 1e4))
        ),
        list(
            type = "FCP_TPA", npc = npc,
            alphaRange = list(v = c(1e-4, 1e4), w = c(1e-4, 1e4))
        )
    )
)

# Save
write.csv(sim$simData[[1]]@X, './data_1.csv', row.names = FALSE)
write.csv(sim$simData[[2]]@X, './data_2.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns[[1]]@X, './true_eigenfunctions_1.csv', row.names = FALSE)
write.csv(sim$trueFuns[[2]]@X, './true_eigenfunctions_2.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[1]]@X, './estim_mu_1.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[2]]@X, './estim_mu_2.csv', row.names = FALSE)
write.csv(fpca$functions[[1]]@X,
          './estim_eigenfunctions_1.csv', row.names = FALSE)
write.csv(fpca$functions[[2]]@X,
          './estim_eigenfunctions_2.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
# ----

### MFPCA 1D + 1D with exponential eigenvalues for variance explained ----
set.seed(42)
sim <-  simMultiFunData(
    type = "split",
    argvals = list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
    M = 20,
    eFunType = "Fourier",
    eValType = "exponential",
    N = N)
pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = 0.95)
pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = 0.95)
fpca <- MFPCA(
    sim$simData, M = pace1$npc + pace2$npc, 
    uniExpansions = list(
        list(type = "uFPCA", pve = 0.95),
        list(type = "uFPCA", pve = 0.95)
    ),
    fit = TRUE)

# Save
write.csv(sim$simData[[1]]@X, './data_1.csv', row.names = FALSE)
write.csv(sim$simData[[2]]@X, './data_2.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns[[1]]@X, './true_eigenfunctions_1.csv', row.names = FALSE)
write.csv(sim$trueFuns[[2]]@X, './true_eigenfunctions_2.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[1]]@X, './estim_mu_1.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[2]]@X, './estim_mu_2.csv', row.names = FALSE)
write.csv(fpca$functions[[1]]@X,
          './estim_eigenfunctions_1.csv', row.names = FALSE)
write.csv(fpca$functions[[2]]@X,
          './estim_eigenfunctions_2.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
write.csv(fpca$fit[[1]]@X, './estim_reconst_1.csv', row.names = FALSE)
write.csv(fpca$fit[[2]]@X, './estim_reconst_2.csv', row.names = FALSE)

# ----

### MFPCA 1D + 1D with linear eigenvalues for variance explained ----
set.seed(42)
sim <-  simMultiFunData(
    type = "split",
    argvals = list(seq(0, 1, 0.01), seq(0, 1, 0.01)),
    M = 20,
    eFunType = "Fourier",
    eValType = "linear",
    N = N)
pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = 0.95)
pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = 0.95)
fpca <- MFPCA(
    sim$simData, M = pace1$npc + pace2$npc, 
    uniExpansions = list(
        list(type = "uFPCA", pve = 0.95),
        list(type = "uFPCA", pve = 0.95)
    ),
    fit = TRUE)

# Save
write.csv(sim$simData[[1]]@X, './data_1.csv', row.names = FALSE)
write.csv(sim$simData[[2]]@X, './data_2.csv', row.names = FALSE)
write.csv(t(sim$trueVals), './true_eigenvalues.csv', row.names = FALSE)
write.csv(sim$trueFuns[[1]]@X, './true_eigenfunctions_1.csv', row.names = FALSE)
write.csv(sim$trueFuns[[2]]@X, './true_eigenfunctions_2.csv', row.names = FALSE)

write.csv(t(fpca$values), './estim_eigenvalues.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[1]]@X, './estim_mu_1.csv', row.names = FALSE)
write.csv(fpca$meanFunction[[2]]@X, './estim_mu_2.csv', row.names = FALSE)
write.csv(fpca$functions[[1]]@X,
          './estim_eigenfunctions_1.csv', row.names = FALSE)
write.csv(fpca$functions[[2]]@X,
          './estim_eigenfunctions_2.csv', row.names = FALSE)
write.csv(fpca$scores, './estim_scores.csv', row.names = FALSE)
write.csv(fpca$fit[[1]]@X, './estim_reconst_1.csv', row.names = FALSE)
write.csv(fpca$fit[[2]]@X, './estim_reconst_2.csv', row.names = FALSE)

# ----
