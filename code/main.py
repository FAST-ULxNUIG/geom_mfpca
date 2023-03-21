################################################################################
# Scripts to run the simulation for the paper "A geometric interpretation of 
# the multivariate functional principal components analysis"
# S. Golovkine - 17/01/2023
################################################################################

# Load packages
import argparse
import json
import multiprocessing
import os
import re
import time

from joblib import Parallel, delayed

from functions import *

# Argument parser for the file
parser = argparse.ArgumentParser(
    description="Simulation parametes.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("out_folder", help="Output folder to store the results")
parser.add_argument(
    "-nsimu", "--number_simulation", default=5,
    help="Number of simulation"
)
parser.add_argument(
    "-nobs", "--number_observation", default=10,
    help="Number of observations"
)
parser.add_argument(
    "-m", "--number_points", default=50,
    help="Number of points per curves"
)
parser.add_argument(
    "-p", "--number_components", default=1,
    help="Number of components to simulate"
)
parser.add_argument(
    "-k", "--number_components_estimate", default=1,
    help="Number of components to estimate for the KL decomposition"
)
args = parser.parse_args()

# Variables
NUM_CORES = multiprocessing.cpu_count()
N_SIMU = int(args.number_simulation)
N = int(args.number_observation)
M = int(args.number_points)
P = int(args.number_components)
n_components = int(args.number_components_estimate)

# Functions
def run_simulation(idx, n_obs, n_points, n_components):
    """Run a simulation"""
    print(f'Simulation {idx}')

    if P == 1:  # Simulate univariate functional data
        basis_name = 'fourier'
        dimension = '2D'
        n_functions = 5

        data = simulate_data(
            n_obs=n_obs,
            n_points=n_points,
            basis_name=basis_name,
            n_functions=n_functions,
            dimension=dimension,
            seed=idx
        )
    else:  # Simulate multivariate functional data
        basis_name = 'wiener'
        n_functions = 10
        data = simulate_data_multivariate(
            n_components=P,
            n_obs=n_obs,
            n_points=n_points,
            basis_name=basis_name,
            n_functions=n_functions,
            seed=idx
        )

    # Perform FPCA covariance
    if P == 1:
        start = time.time()
        fpca_cov = ufpca_covariance(data, n_components)
        time_cov = time.time() - start

        # Perform FPCA inner-product
        start = time.time()
        fpca_inn = ufpca_inner_product(data, n_components)
        time_inn = time.time() - start
    else:
        start_cov = time.time()
        fpca_cov = mfpca_covariance(data, n_components)
        time_cov = time.time() - start_cov

        # Perform FPCA inner-product
        start_inn = time.time()
        fpca_inn = mfpca_inner_product(data, n_components)
        time_inn = time.time() - start_inn

    # Compute scores
    scores_cov = compute_scores(data, fpca_cov)
    scores_inn = compute_scores(data, fpca_inn)

    # Compute reconstruction
    data_f_cov = compute_reconstruction(fpca_cov, scores_cov)
    data_f_inn = compute_reconstruction(fpca_inn, scores_inn)

    # Compute MISE
    if P == 1:
        mise_cov = MISE_2D(data, data_f_cov)
        mise_inn = MISE_2D(data, data_f_inn)
    else:
        mise_cov = MISE(data, data_f_cov)
        mise_inn = MISE(data, data_f_inn)

    return {
        'time_cov': time_cov,
        'time_inn': time_inn,
        'mise_cov': mise_cov,
        'mise_inn': mise_inn,
        'n_comp_cov': fpca_cov.eigenfunctions.n_obs,
        'n_comp_inn': fpca_inn.eigenfunctions.n_obs
    }


# Run
if __name__ == "__main__":
    print("Run the simulation")
	
    start = time.time()
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(run_simulation)(idx, N, M, n_components)
        for idx in range(N_SIMU)
    )
    print(f'{time.time() - start}')
    
    if not os.path.exists(f"{args.out_folder}/P{P}"):
        os.makedirs(f"{args.out_folder}/P{P}")

    NAME = f'results_{N}_{M}_{P}_{N_SIMU}.json'
    with open(f'{args.out_folder}/P{P}/{NAME}', 'w') as outfile:
        json.dump(results, outfile)
