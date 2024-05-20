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
from FDApy.preprocessing.smoothing.psplines import PSplines
from FDApy.simulation.karhunen import (
    _initialize_centers,
    _initialize_clusters_std,
    _make_coef
)
from FDApy.simulation.simulation import (
    _add_noise_univariate_data,
    _sparsify_univariate_data
)

# Variables

# Functions
def simulate_data(
    n_obs, n_points, noise_variance=0, percentages=[1, 1], epsilon=0.1, seed=42,
):
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
        'input_dim_0': np.linspace(0, 1, n_points[0]),
        'input_dim_1': np.linspace(0, 0.5, n_points[1]),
    })
    basis_1 = Basis(
        name=['fourier', 'fourier'],
        n_functions=[5, 5],
        argvals=argvals
    )
    
    argvals = DenseArgvals({
        'input_dim_0': np.linspace(-1, 1, n_points[2])
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

    # Generate noise data
    data_noisy = MultivariateFunctionalData([
        _add_noise_univariate_data(
            data_univariate, noise_variance=noise_variance, rnorm=rng.normal
        ) for data_univariate in data.data
    ])

    # Generate sparse data
    data_sparse = MultivariateFunctionalData([
        _sparsify_univariate_data(
            data_univariate,
            percentage=percentage,
            epsilon=epsilon,
            runif=rng.uniform,
            rchoice=rng.choice
        ) for (data_univariate, percentage) in zip(data.data, percentages)
    ])

    return {
        'data': data,
        'data_noisy': data_noisy,
        'data_sparse': data_sparse,
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
    
    new_values = np.zeros_like(data.values)
    for idx, (dd, dd_ref) in enumerate(zip(data, data_reference)):
        norm_pos = np.linalg.norm(dd.values + dd_ref.values)
        norm_neg = np.linalg.norm(dd.values - dd_ref.values)
    
        sign = -1 if norm_pos < norm_neg else 1
        new_values[idx, :] = sign * dd.values.squeeze()
    return DenseFunctionalData(data.argvals, new_values)


def flip_multivariate(
    data: MultivariateFunctionalData,
    data_reference: MultivariateFunctionalData
):
    """Flip data if they have opposite sign."""    
    new_data = data.n_functional * [None]
    for idx, (dd, dd_ref) in enumerate(zip(data.data, data_reference.data)):
        new_data[idx] = flip(dd, dd_ref)
    return MultivariateFunctionalData(new_data)


def AE(data, data_reference):
    """Compute AE between two eigenvalues sets.

    Parameters
    ----------
    data: array
        Estimated eigenvalues.
    data_reference: array
        True eigenvalues.

    Returns
    -------
    np.NDArray[np.float_], shape=(n_obs,)
        An array containing the ISE between curve

    """
    return (data_reference - data)**2 / data_reference**2


def ISE(
    data: MultivariateFunctionalData,
    data_reference: MultivariateFunctionalData
):
    """Compute ISE between two univariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data: MultivariateFunctionalData
        Estimated curves.
    data_reference: MultivariateFunctionalData
        True curves.

    Returns
    -------
    np.NDArray[np.float_], shape=(n_obs,)
        An array containing the ISE between curve

    """
    # Flip the data
    new_data = flip_multivariate(data, data_reference)

    results = np.zeros(new_data.n_obs)
    for idx, (dd, dd_ref) in enumerate(zip(new_data, data_reference)):
        temp = sum(
            (ddd - ddd_ref).norm(squared=True)
            for ddd, ddd_ref in zip(dd.data, dd_ref.data)
        )
        results[idx] = temp
    return results


def MRSE(
    data: MultivariateFunctionalData,
    data_reference: MultivariateFunctionalData
):
    """Compute MRSE between two univariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data: MultivariateFunctionalData
        Estimated curves.
    data_reference: MultivariateFunctionalData
        True curves.

    Returns
    -------
    np.NDArray[np.float_], shape=(n_obs,)
        An array containing the ISE between curve

    """

    results = np.zeros(data.n_obs)
    for idx, (dd, dd_ref) in enumerate(zip(data, data_reference)):
        norm_diff = [
            (ddd - ddd_ref).norm(squared=True) / ddd_ref.norm(squared=True)
            for ddd, ddd_ref in zip(dd.data, dd_ref.data)
        ]
        results[idx] = np.sum(norm_diff)
    return np.mean(results)


def cross_validation(x, y, penalties):
    """Do CV for the estimation of penalty in PSplines model.

    Parameters
    ----------
    x: array
        Sample points.
    y: array
        Observation points.
    penalties: array
        Array of penalties.

    Returns
    -------
    Tuple[float]
        The penalty that minimizes CV.

    """
    CV = np.zeros(len(penalties))
    for idx, penalty in enumerate(penalties):
        ps = PSplines()
        ps.fit(x=x, y=y, penalty=penalty)
        H = ps.diagnostics['hat_matrix']
        R = (y - ps.y_hat) / (1 - H)
        CV[idx] = np.sqrt(np.mean(np.power(R, 2)))
    return penalties[np.argmin(CV)]
