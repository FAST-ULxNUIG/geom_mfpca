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
