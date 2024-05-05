################################################################################
# Utility functions for the simulations of univariate functional data
# S. Golovkine - 05/04/2024
################################################################################

# Load packages
import numpy as np

from FDApy.representation.basis import Basis, MultivariateBasis
from FDApy.representation.argvals import DenseArgvals
from FDApy.representation.values import DenseValues
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    BasisFunctionalData,
    MultivariateFunctionalData
)
from FDApy.simulation.karhunen import (
    _initialize_centers,
    _initialize_clusters_std,
    _make_coef
)
from FDApy.simulation.simulation import (
    _add_noise_univariate_data
)

# Variables

# Functions
def simulate_data(n_obs, noise_variance, seed):
    """Simulation function.

    Parameters
    ----------
    n_obs: int
        Number of observations to simulated.
    noise_variance: float
        Variance of the noise.
    seed: int
        A seed for reproducibility.

    Returns
    -------
    KarhunenLoeve
        The simulated dataset.

    """
    # Set the RNG
    rng = np.random.default_rng(seed)

    # Basis definition
    argvals = DenseArgvals({
        'input_dim_0': np.linspace(0, 1, 101),
        'input_dim_1': np.linspace(0, 0.5, 51),
    })
    basis_1 = Basis(
        name=['fourier', 'fourier'],
        n_functions=[5, 5],
        argvals=argvals
    )
    
    argvals = DenseArgvals({
        'input_dim_0': np.linspace(-1, 1, 201)
    })
    basis_2 = Basis(
        name='legendre',
        n_functions=25,
        argvals=argvals,
    )

    # Normalization of the basis
    u1 = rng.uniform(0.2, 0.8, 1)
    u2 = rng.uniform(0.2, 0.8, 1)
    alpha = u1 / (u1 + u2)

    # Define multivariate basis
    basis = MultivariateBasis(
        name='given',
        argvals=[basis_1.argvals, basis_2.argvals],
        values=[
            np.sqrt(alpha) * basis_1.values,
            np.sqrt(1 - alpha) * basis_2.values
        ]
    )   

    # Initialize parameters
    n_features = basis.n_obs
    centers = _initialize_centers(n_features, 1, None)
    clusters_std = _initialize_clusters_std(n_features, 1, 'exponential')

    # Generate coefficients
    rnorm = rng.multivariate_normal
    coef, _ = _make_coef(n_obs, n_features, centers, clusters_std, rnorm)

    # Generate data
    data = MultivariateFunctionalData([
        BasisFunctionalData(basis=basis_univariate, coefficients=coef).to_grid()
        for basis_univariate in basis.data
    ])

    # Generate noise day
    data_noisy = MultivariateFunctionalData([
        _add_noise_univariate_data(
            data_univariate, noise_variance=noise_variance, rnorm=rng.normal
        ) for data_univariate in data.data
    ])

    return {
        'data': data,
        'data_noisy': data_noisy,
        'basis': basis,
        'eigenvalues': clusters_std[:, 0]
    }

def flip(
    data: DenseFunctionalData,
    data_reference: DenseFunctionalData
):
    """Flip data if they have opposite sign."""
    if isinstance(data, BasisFunctionalData):
        data = data.to_grid()
    if isinstance(data_reference, BasisFunctionalData):
        data_reference = data_reference.to_grid()
    norm_pos = np.linalg.norm(data.values + data_reference.values)
    norm_neg = np.linalg.norm(data.values - data_reference.values)
    
    sign = -1 if norm_pos < norm_neg else 1
    return DenseFunctionalData(data.argvals, sign * data.values)


def flip_multi(
    data: MultivariateFunctionalData,
    data_reference: MultivariateFunctionalData
):
    """Flip data if they have opposite sign."""
    n_obs = data.n_obs
    
    data_list = data.n_functional * [None]
    for idx, (d, d_ref) in enumerate(zip(data.data, data_reference[:n_obs].data)):
        data_list[idx] = flip(d, d_ref)
    return MultivariateFunctionalData(data_list)


def ISE(data_true, data_recons, n_estim=None):
    """Compute MISE between two univariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data_true: DenseFunctionalData
        Dataset of functional data.
    data_recons: DenseFunctionalData
        Data of functionalData.
    n_estim: np.int64
        Number of functions to estimate

    Returns
    -------
    float
        The mean integrated squared error between the two datasets.

    """
    if isinstance(data_true, DenseFunctionalData):
        data_true = MultivariateFunctionalData([data_true])
    if isinstance(data_recons, DenseFunctionalData):
        data_recons = MultivariateFunctionalData([data_recons])
    if n_estim is None:
        n_estim = data_true.n_obs
    
    res = n_estim * [None]
    for idx in np.arange(n_estim):
        data_t = data_true.get_obs(idx)
        data_f = flip_multi(data_recons.get_obs(idx), data_true.get_obs(idx))

        ise = data_true.n_functional * [None]
        for p in np.arange(data_t.n_functional):
            values_true = data_t[p].values.squeeze()
            values_recons = data_f[p].values.squeeze()
            diff_squared = np.power(values_true - values_recons, 2)
            if len(values_true.shape) == 2:
                int_along_y = np.trapz(y=diff_squared, x=data_t[p].argvals['input_dim_1'])
                ise[p] = np.trapz(y=int_along_y, x=data_t[p].argvals['input_dim_0'])
            else:
                ise[p] = np.trapz(y=diff_squared, x=data_t[p].argvals['input_dim_0'])
        res[idx] = np.sum(ise)
    return res


def MISE(data_true, data_recons, n_estim=None):
    """Compute MISE between two multivariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data_true: MultivariateFunctionalData
        Dataset of functional data.
    data_recons: MultivariateFunctionalData
        Data of functionalData.
    n_estim: np.int64
        Number of functions to estimate

    Returns
    -------
    float
        The mean integrated squared error between the two datasets.

    """
    return np.mean(ISE(data_true, data_recons, n_estim))


def logAE(eigenvalues_true, eigenvalues_estim, n_estim):
    """Compute log-AE between two sets of eigenvalues.

    Parameters
    ----------
    eigenvalues_true: npt.NDArray
        Array of eigenvalues.
    eigenvalues_estim: npt.NDArray
        Array of eigenvalues.
    n_estim: np.int64
        Number of eigenvalues to estimate

    Returns
    -------
    npt.NDArray
        The log-AE for each eigenvalues estimation.

    """
    res = n_estim * [None]
    for idx in np.arange(n_estim):
        res[idx] = np.abs(eigenvalues_true[idx] - eigenvalues_estim[idx])
    return np.log(res)
