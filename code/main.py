################################################################################
# Scripts to run the simulation for the paper "On the use of the Gram matrix 
# for multivariate functional principal analysis"
# S. Golovkine - 05/04/2024
################################################################################

# Load packages
import argparse
import itertools
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
    "-nobs", "--number_observation", default=10, help="Number of points"
)
parser.add_argument(
    "-npoints", "--number_points", nargs='+', type=int,
    default=[101, 51, 201], help="Number of observations"
)
parser.add_argument(
    "-noise", "--noise_variance", default=0.25,
    help="Variance of the noise"
)
parser.add_argument(
    "-percentages", "--percentages", nargs='+', type=float,
    default=[0.1, 0.8], help="Percentages"
)
parser.add_argument(
    "-epsilon", "--epsilon", type=float, default=0.1, help="Epsilon"
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
n_points = args.number_points
noise_variance = float(args.noise_variance)
percentages = args.percentages
epsilon = float(args.epsilon)
n_components = int(args.number_components_estimate)

# Functions
def run_simulation_no_noise(
    idx, n_obs, n_points, noise_variance, percentages, n_components, PATH
):
    """Run a simulation"""
    print(f'Simulation {idx}')

    simulation = simulate_data(
        n_obs=n_obs,
        n_points=n_points,
        noise_variance=noise_variance,
        percentages=percentages,
        epsilon=0,
        seed=idx
    )

    data = simulation['data']
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
        {'method': 'PSplines', 'penalty': (0, 0)},
        {'method': 'PSplines', 'penalty': (0)}
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
    scores_covariance = mfpca_covariance.transform()
    scores_psplines = mfpca_psplines.transform()
    scores_gram = mfpca_gram.transform(method='InnPro')


    # Reconstructing the data
    data_recons_covariance = mfpca_covariance.inverse_transform(
        scores_covariance
    )
    data_recons_psplines = mfpca_psplines.inverse_transform(scores_psplines)
    data_recons_gram = mfpca_gram.inverse_transform(scores_gram)

    # Results
    true_eigenvalues = eigenvalues[:n_components]
    true_eigenfunctions = eigenfunctions[:n_components]
    errors = {
        # Eigenvalues
        'AE_cov': AE(mfpca_covariance.eigenvalues, true_eigenvalues),
        'AE_psplines': AE(mfpca_psplines.eigenvalues, true_eigenvalues),
        'AE_gram': AE(mfpca_gram.eigenvalues, true_eigenvalues),
        # Eigenfunctions
        'ISE_cov': ISE(mfpca_covariance.eigenfunctions, true_eigenfunctions),
        'ISE_psplines': ISE(mfpca_psplines.eigenfunctions, true_eigenfunctions),
        'ISE_gram': ISE(mfpca_gram.eigenfunctions, true_eigenfunctions),
        # Curves reconstruction
        'MRSE_cov': MRSE(data_recons_covariance, data),
        'MRSE_psplines': MRSE(data_recons_psplines, data),
        'MRSE_gram': MRSE(data_recons_gram, data)
    }

    ##########################################################################
    results = {
        # Time
        'time_covariance': time_covariance,
        'time_psplines': time_psplines,
        'time_gram': time_gram,
        # Results
        'errors': errors
    }

    NAME = f'{PATH}/simulation_{idx}.pkl'
    with open(NAME, "wb") as f:
        pickle.dump(results, f)

    ##########################################################################

def run_simulation_noise(
    idx, n_obs, n_points, noise_variance, percentages, n_components, PATH
):
    """Run a simulation"""
    print(f'Simulation {idx}')

    simulation = simulate_data(
        n_obs=n_obs,
        n_points=n_points,
        noise_variance=noise_variance,
        percentages=percentages,
        epsilon=0,
        seed=idx
    )

    data = simulation['data']
    data_noisy = simulation['data_noisy']
    eigenfunctions = simulation['basis']
    eigenvalues = simulation['eigenvalues']

    ##########################################################################
    print(f"Run MFPCA for data with noise (simulation {idx})")

    # CV
    n_samples = 10
    random_samples = data_noisy[np.random.choice(data.n_obs, n_samples)]
    penalties_1d = np.float_power(10, np.arange(-3, 3, .1))
    penalties_2d = list(
        itertools.product(np.float_power(10, np.arange(-2, 2, .25)), repeat=2)
    )

    # 1D P-Splines smoothing using CV
    CVS = []
    for curve in random_samples.data[1]:
        x = curve.argvals['input_dim_0']
        y = curve.values.squeeze()
        CV = cross_validation(x, y, penalties_1d)
        CVS.append(CV)
    penalty_CV_1d = np.median(CVS, axis=0)


    # 2D P-Splines smoothing using CV
    CVS = []
    for curve in random_samples.data[0]:
        x = [curve.argvals['input_dim_0'], curve.argvals['input_dim_1']]
        y = curve.values.squeeze()
        CV = cross_validation(x, y, penalties_2d)
        CVS.append(CV)
    penalty_CV_2d = np.median(CVS, axis=0)

    # 1D P-Splines smoothing using CV for mean
    data_mean = data_noisy.data[1].mean(method_smoothing=None)

    x = data_mean.argvals['input_dim_0']
    y = data_mean.values.squeeze()
    penalty_mean = cross_validation(x, y, penalties_1d)

    # 2D P-Splines smoothing using CV for covariance
    data_cov = data_noisy.data[1].covariance(method_smoothing=None)

    x = [data_cov.argvals['input_dim_0'], data_cov.argvals['input_dim_1']]
    y = data_cov.values.squeeze()
    penalty_cov = cross_validation(x, y, penalties_2d)


    # Run MFPCA with FCPTPA and UFPCA
    univariate_expansions = [
        {
            'method': 'FCPTPA',
            'n_components': 20
        },
        {
            'method': 'UFPCA',
            'n_components': 15,
            'method_smoothing': 'PS',
            'kwargs_mean': {'penalty': penalty_mean},
            'kwargs_covariance': {'penalty': penalty_cov}
        }
    ]
    mfpca_covariance = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_covariance.fit(data_noisy)
    time_covariance = time.process_time() - start


    # Run MFPCA with PSplines
    univariate_expansions = [
        {'method': 'PSplines', 'penalty': penalty_CV_2d},
        {'method': 'PSplines', 'penalty': penalty_CV_1d}
    ]
    mfpca_psplines = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_psplines.fit(data_noisy)
    time_psplines = time.process_time() - start


    # Run MFPCA with Gram matrix
    univariate_expansions = [
        {'method': 'PS', 'penalty': penalty_CV_2d},
        {'method': 'PS', 'penalty': penalty_CV_1d}
    ]
    data_smooth = MultivariateFunctionalData([
        fdata.smooth(
            method=univ_expansion['method'],
            penalty=univ_expansion['penalty']
        ) for (fdata, univ_expansion) in zip(
            data_noisy.data, univariate_expansions
        )
    ])

    mfpca_gram = MFPCA(n_components=n_components, method='inner-product')
    start = time.process_time()
    mfpca_gram.fit(data_smooth)
    time_gram = time.process_time() - start


    # Estimate scores
    scores_covariance = mfpca_covariance.transform()
    scores_psplines = mfpca_psplines.transform()
    scores_gram = mfpca_gram.transform(method='InnPro')


    # Reconstructing the data
    data_recons_covariance = mfpca_covariance.inverse_transform(
        scores_covariance
    )
    data_recons_psplines = mfpca_psplines.inverse_transform(
        scores_psplines
    )
    data_recons_gram = mfpca_gram.inverse_transform(
        scores_gram
    )

    # Results
    true_eigenvalues = eigenvalues[:n_components]
    true_eigenfunctions = eigenfunctions[:n_components]
    errors = {
        # Eigenvalues
        'AE_cov': AE(mfpca_covariance.eigenvalues, true_eigenvalues),
        'AE_psplines': AE(mfpca_psplines.eigenvalues, true_eigenvalues),
        'AE_gram': AE(mfpca_gram.eigenvalues, true_eigenvalues),
        # Eigenfunctions
        'ISE_cov': ISE(mfpca_covariance.eigenfunctions, true_eigenfunctions),
        'ISE_psplines': ISE(mfpca_psplines.eigenfunctions, true_eigenfunctions),
        'ISE_gram': ISE(mfpca_gram.eigenfunctions, true_eigenfunctions),
        # Curves reconstruction
        'MRSE_cov': MRSE(data_recons_covariance, data),
        'MRSE_psplines': MRSE(data_recons_psplines, data),
        'MRSE_gram': MRSE(data_recons_gram, data)
    }

    ##########################################################################

    results = {
        # Time
        'time_covariance': time_covariance,
        'time_psplines': time_psplines,
        'time_gram': time_gram,
        # Results
        'errors': errors
    }

    NAME = f'{PATH}/simulation_{idx}.pkl'
    with open(NAME, "wb") as f:
        pickle.dump(results, f)


def run_simulation_sparse(
    idx, n_obs, n_points, noise_variance, percentages,
    epsilon, n_components, PATH
):
    """Run a simulation"""
    print(f'Simulation {idx}')

    simulation = simulate_data(
        n_obs=n_obs,
        n_points=n_points,
        noise_variance=noise_variance,
        percentages=percentages,
        epsilon=epsilon,
        seed=idx
    )

    data = simulation['data']
    data_sparse = simulation['data_sparse']
    eigenfunctions = simulation['basis']
    eigenvalues = simulation['eigenvalues']

    ##########################################################################
    print(f"Run MFPCA for sparse data (simulation {idx})")


    # Run MFPCA with FCPTPA and UFPCA
    univariate_expansions = [
        {
            'method': 'FCPTPA',
            'n_components': 20
        },
        {
            'method': 'UFPCA',
            'n_components': 15,
            'method_smoothing': 'PS',
            'kwargs_mean': {'penalty': 0},
            'kwargs_covariance': {'penalty': (0, 0)}
        }
    ]
    mfpca_covariance = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_covariance.fit(data_sparse, method_smoothing='PS')
    time_covariance = time.process_time() - start


    # Run MFPCA with PSplines
    univariate_expansions = [
        {'method': 'PSplines', 'penalty': (0, 0)},
        {'method': 'PSplines', 'penalty': 0}
    ]
    mfpca_psplines = MFPCA(
        n_components=n_components,
        method='covariance',
        univariate_expansions=univariate_expansions
    )
    start = time.process_time()
    mfpca_psplines.fit(data_sparse, method_smoothing='PS')
    time_psplines = time.process_time() - start


    # Run MFPCA with Gram matrix
    mfpca_gram = MFPCA(n_components=n_components, method='inner-product')
    start = time.process_time()
    mfpca_gram.fit(data_sparse, method_smoothing='PS')
    time_gram = time.process_time() - start


    # Estimate scores
    scores_covariance = mfpca_covariance.transform()
    scores_psplines = mfpca_psplines.transform()
    scores_gram = mfpca_gram.transform(method='InnPro')


    # Reconstructing the data
    data_recons_covariance = mfpca_covariance.inverse_transform(
        scores_covariance
    )
    data_recons_psplines = mfpca_psplines.inverse_transform(
        scores_psplines
    )
    data_recons_gram = mfpca_gram.inverse_transform(
        scores_gram
    )

    # Results
    true_eigenvalues = eigenvalues[:n_components]
    true_eigenfunctions = eigenfunctions[:n_components]
    errors = {
        # Eigenvalues
        'AE_cov': AE(mfpca_covariance.eigenvalues, true_eigenvalues),
        'AE_psplines': AE(mfpca_psplines.eigenvalues, true_eigenvalues),
        'AE_gram': AE(mfpca_gram.eigenvalues, true_eigenvalues),
        # Eigenfunctions
        'ISE_cov': ISE(mfpca_covariance.eigenfunctions, true_eigenfunctions),
        'ISE_psplines': ISE(mfpca_psplines.eigenfunctions, true_eigenfunctions),
        'ISE_gram': ISE(mfpca_gram.eigenfunctions, true_eigenfunctions),
        # Curves reconstruction
        'MRSE_cov': MRSE(data_recons_covariance, data),
        'MRSE_psplines': MRSE(data_recons_psplines, data),
        'MRSE_gram': MRSE(data_recons_gram, data)
    }

    ##########################################################################

    results = {
        # Time
        'time_covariance': time_covariance,
        'time_psplines': time_psplines,
        'time_gram': time_gram,
        # Results
        'errors': errors
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
    POINTS = '-'.join(str(x) for x in n_points)
    
    if noise_variance == 0 and epsilon == 0:    
        if not os.path.exists(f"{PATH}/N{N}_M{POINTS}"):
            os.makedirs(f"{PATH}/N{N}_M{POINTS}")

        Parallel(n_jobs=NUM_CORES)(
            delayed(run_simulation_no_noise)(
                idx, N, n_points, noise_variance, percentages, n_components,
                PATH + "/" + f"N{N}_M{POINTS}"
            ) for idx in range(N_SIMU)
        )
    elif noise_variance != 0 and epsilon == 0:
        if not os.path.exists(f"{PATH}/N{N}_M{POINTS}_S{noise_variance}"):
            os.makedirs(f"{PATH}/N{N}_M{POINTS}_S{noise_variance}")

        Parallel(n_jobs=NUM_CORES)(
            delayed(run_simulation_noise)(
                idx, N, n_points, noise_variance, percentages, n_components,
                PATH + "/" + f"N{N}_M{POINTS}_S{noise_variance}"
            ) for idx in range(N_SIMU)
        )
    else:
        if not os.path.exists(f"{PATH}/N{N}_M{POINTS}_eps{epsilon}"):
            os.makedirs(f"{PATH}/N{N}_M{POINTS}_eps{epsilon}")

        Parallel(n_jobs=NUM_CORES)(
            delayed(run_simulation_sparse)(
                idx, N, n_points, noise_variance,
                percentages, epsilon, n_components,
                PATH + "/" + f"N{N}_M{POINTS}_eps{epsilon}"
            ) for idx in range(N_SIMU)
        )
