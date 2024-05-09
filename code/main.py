################################################################################
# Scripts to run the simulation for the paper "On the use of the Gram matrix 
# for multivariate functional principal analysis"
# S. Golovkine - 05/04/2024
################################################################################

# Load packages
import argparse
import pickle
import multiprocessing
import os
import time

from joblib import Parallel, delayed

from FDApy.representation.functional_data import MultivariateFunctionalData
from FDApy.preprocessing import MFPCA
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
    "-noise", "--noise_variance", default=0.25,
    help="Variance of the noise"
)
parser.add_argument(
    "-percentages", "--percentages", nargs='+', type=float,
    help="Percentages"
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
noise_variance = float(args.noise_variance)
percentages = args.percentages
n_components = int(args.number_components_estimate)

# Functions
def run_simulation(idx, n_obs, noise_variance, percentages, n_components, PATH):
    """Run a simulation"""
    print(f'Simulation {idx}')

    simulation = simulate_data(
        n_obs=n_obs,
        noise_variance=noise_variance,
        percentages=percentages,
        seed=idx
    )

    data = simulation['data']
    data_noisy = simulation['data_noisy']
    data_sparse = simulation['data_sparse']
    eigenfunctions = simulation['basis']
    eigenvalues = simulation['eigenvalues']

    ##########################################################################
    print(f"Run MFPCA for data without noise (simulation {idx})")

    # Run MFPCA with FCPTPA and UFPCA
    univariate_expansions = [
        {'method': 'FCPTPA', 'n_components': 20},
        {'method': 'UFPCA', 'n_components': 15}
    ]
    mfpca_covariance = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_covariance.fit(data)
    time_covariance = time.process_time() - start


    # Run MFPCA with PSplines
    univariate_expansions = [
        {'method': 'PSplines', 'penalty': [0, 0]},
        {'method': 'PSplines', 'penalty': [0]}
    ]
    mfpca_psplines = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_psplines.fit(data)
    time_psplines = time.process_time() - start


    # Run MFPCA with Gram matrix
    mfpca_gram = MFPCA(n_components=n_components, method='inner-product')
    start = time.process_time()
    mfpca_gram.fit(data)
    time_gram = time.process_time() - start


    # Estimate scores
    scores_covariance = mfpca_covariance.transform(data)
    scores_psplines = mfpca_psplines.transform(data)
    scores_gram = mfpca_gram.transform(method='InnPro')


    # Reconstructing the data
    data_recons_covariance = mfpca_covariance.inverse_transform(
        scores_covariance
    )
    data_recons_psplines = mfpca_psplines.inverse_transform(scores_psplines)
    data_recons_gram = mfpca_gram.inverse_transform(scores_gram)

    ##########################################################################

    ##########################################################################
    print(f"Run MFPCA for data with noise (simulation {idx})")

    # Run MFPCA with FCPTPA and UFPCA
    univariate_expansions = [
        {'method': 'FCPTPA', 'n_components': 20},
        {'method': 'UFPCA', 'n_components': 15}
    ]
    mfpca_covariance_noise = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    mfpca_covariance_noise.fit(data_noisy)


    # Run MFPCA with PSplines
    univariate_expansions = [
        {'method': 'PSplines', 'penalty': [1, 1]},
        {'method': 'PSplines', 'penalty': [1]}
    ]
    mfpca_psplines_noise = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    mfpca_psplines_noise.fit(data_noisy)


    # Run MFPCA with Gram matrix
    mfpca_gram_noise = MFPCA(n_components=n_components, method='inner-product')
    mfpca_gram_noise.fit(data_noisy)


    # Estimate scores
    scores_covariance_noise = mfpca_covariance_noise.transform(data_noisy)
    scores_psplines_noise = mfpca_psplines_noise.transform(data_noisy)
    scores_gram_noise = mfpca_gram_noise.transform(method='InnPro')


    # Reconstructing the data
    data_recons_covariance_noise = mfpca_covariance_noise.inverse_transform(
        scores_covariance_noise
    )
    data_recons_psplines_noise = mfpca_psplines_noise.inverse_transform(
        scores_psplines_noise
    )
    data_recons_gram_noise = mfpca_gram_noise.inverse_transform(
        scores_gram_noise
    )

    ##########################################################################

    results = {
        # Time
        'time_covariance': time_covariance,
        'time_psplines': time_psplines,
        'time_gram': time_gram,
        # Data
        'data_true': data,
        'eigenfunctions_true': eigenfunctions,
        'eigenvalues_true': eigenvalues,
        # Estimation data no noise
        'mfpca_covariance': mfpca_covariance,
        'mfpca_psplines': mfpca_psplines,
        'mfpca_gram': mfpca_gram,
        'data_recons_covariance': data_recons_covariance,
        'data_recons_psplines': data_recons_psplines,
        'data_recons_gram': data_recons_gram,
        # Estimation data noise
        'mfpca_covariance_noise': mfpca_covariance_noise,
        'mfpca_psplines_noise': mfpca_psplines_noise,
        'mfpca_gram_noise': mfpca_gram_noise,
        'data_recons_covariance_noise': data_recons_covariance_noise,
        'data_recons_psplines_noise': data_recons_psplines_noise,
        'data_recons_gram_noise': data_recons_gram_noise
    }

    NAME = f'{PATH}/simulation_{idx}.pkl'
    with open(NAME, "wb") as f:
        pickle.dump(results, f)


# Run
if __name__ == "__main__":
    print("Run the simulation")
	
    if not os.path.exists(f"{args.out_folder}"):
        os.makedirs(f"{args.out_folder}")

    PATH = args.out_folder

    start = time.process_time()
    Parallel(n_jobs=NUM_CORES)(
        delayed(run_simulation)(
            idx, N, noise_variance, percentages, n_components, PATH
        ) for idx in range(N_SIMU)
    )
    print(f'{time.process_time() - start}')
