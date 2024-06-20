#-------------------------------------------------------------------------
# See attached, a very brief simulation. It seems to show that choosing a
# univariate cut-off within each dimension (e.g., 95%), tends to overestimate the
# final amount of variance – the sum of the final eigenvalues is larger than the
# sum of the true eigenvalues.
# 
# I did some reading -- Happ and Greven considered the effect of pve only on
# eigenfunction estimation. I’ve attached the relevant pages from their 
# supplementary material. I’m not sure they considered eigenvalues  or the total
# variation after. Moreover, I think (but not sure) they assume in their simulation
# that M is known where they say min{M_1 + M_2, M}; whereas in practice, it is
# obviously unknown. But maybe they’re just referring to M as the total number of
# sample eigenfunctions with non-zero eigenvalues (rather than the true number of
# multivariate eiegenfunctions.

# -------------------------------------------------------------------------

library(MFPCA) # CRAN v1.3-9
set.seed(1)
# -------------------------------------------------------------------------
N_sim <- 200
evals_list_01 <- evals_list_02 <- vector("list",length = N_sim)
true_vals_01 <- simMultiFunData(
    type = "weighted", 
    argvals = list(list(seq(0,1,0.01)), list(seq(0,1,0.01))),
    M = list(20, 20), N = 1,
    eFunType = list("Fourier", "Wiener"),
    eValType = "linear")$trueVals

sum(cumsum(true_vals_01) / sum(true_vals_01) < 0.95) + 1

true_vals_02 <- simMultiFunData(
    type = "weighted",
    argvals = list(list(seq(0,1,0.01)), list(seq(0,1,0.01))),
    M = list(20, 20), N = 1,
    eFunType = list("Fourier", "Wiener"),
    eValType = "exponential")$trueVals

sum(cumsum(true_vals_02) / sum(true_vals_02) < 0.95) + 1

# Step 1 -- Linearly Decreasing Eigenvalues -------------------------------
for (i in seq_len(N_sim)) {
  print(paste("Iteration", i))
  # sim <-  simMultiFunData(
  #     type = "weighted",
  #     argvals = list(list(seq(0,1,0.01)), list(seq(0,1,0.01))),
  #     M = list(20, 20),
  #     eFunType = list("Fourier", "Wiener"),
  #     eValType = "linear",
  #     N = 500)
  sim <-  simMultiFunData(
        type = "split",
        argvals = list(seq(0,1,0.01), seq(0,1,0.01)),
        M = 20,
        eFunType = "Fourier",
        eValType = "linear",
        N = 500)
  pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], pve = 0.99999999999999)
  pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], pve = 0.99999999999999)
  # MFPCA based on univariate FPCA
  mf_list <- multiFunData(list(pace1$fit, pace2$fit))
  MFPCA_est <- MFPCA(
      sim$simData,
      M = pace1$npc + pace2$npc, 
      uniExpansions = list(
          list(type = "uFPCA", pve = 0.99999999999999),
          list(type = "uFPCA", pve = 0.99999999999999)),
      fit = TRUE)
  evals_list_01[[i]] <- MFPCA_est$values
}


# Step 2 -- Exponentially Decreasing Eigenvalues -------------------------------
for (i in seq_len(N_sim)) {
  print(paste("Iteration", i))
  sim <-  simMultiFunData(
      type = "weighted",
      argvals = list(list(seq(0,1,0.01)), list(seq(0,1,0.01))),
      M = list(20, 20),
      eFunType = list("Fourier", "Wiener"),
      eValType = "exponential",
      N = 500)
  pace1 <- MFPCA::PACE(funDataObject = sim$simData[[1]], npc = 20)
  pace2 <- MFPCA::PACE(funDataObject = sim$simData[[2]], npc = 20)
  # MFPCA based on univariate FPCA
  #mf_list <- multiFunData(list(pace1$fit, pace2$fit))
  MFPCA_est <- MFPCA(
      sim$simData, M = pace1$npc + pace2$npc, 
      uniExpansions = list(
          list(type = "uFPCA", npc = 20), 
          list(type = "uFPCA", npc = 20)),
      fit = TRUE)
  evals_list_02[[i]] <- MFPCA_est$values
}


plot(MFPCA_est$values)
lines(sim$trueVals)

par(mfrow = c(1, 2))
boxplot(sapply(evals_list_01, sum) / sum(true_vals_01))
abline(h = 0.95)
boxplot(sapply(evals_list_02, sum) / sum(true_vals_02))
abline(h = 0.95)



# (should also check FPCs estimation.)


