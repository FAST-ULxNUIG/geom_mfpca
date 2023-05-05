################################################################################
# Utility functions for the simulations of univariate functional data
# S. Golovkine - 17/01/2023
################################################################################

# Load packages
import numpy as np

from FDApy.preprocessing.dim_reduction.fpca import UFPCA, MFPCA
from FDApy.preprocessing.dim_reduction.fcp_tpa import FCPTPA
from FDApy.representation.basis import Basis, MultivariateBasis
from FDApy.representation.functional_data import (
    DenseFunctionalData,
    MultivariateFunctionalData
)
from FDApy.simulation.karhunen import KarhunenLoeve

# Variables

# Functions
def make_diff_op(degree, dim):
    if degree == 0:
        return np.eye(dim)
    else:
        return np.diff(make_diff_op(degree - 1, dim), axis=0)


def simulate_data(
    n_obs,
    n_points,
    basis_name,
    n_functions,
    dimension,
    seed
):
    """Simulation function for univariate functional data.

    Parameters
    ----------
    n_obs: int
        Number of observations to simulated.
    n_points: int
        Number of points used to simulate the data. For 2D-data, the number
        of generated points is n_points * n_points.
    basis_name: str
        Name of the basis to used.
    n_functions: int
        Number of basis functions to used.
    dimension: str
        Dimension of the data (1D or 2D).
    seed: int
        A seed for reproducibility.

    Returns
    -------
    KarhunenLoeve
        The simulated dataset.

    """
    # Define the basis
    argvals = np.linspace(0, 1, n_points)
    basis = Basis(
        name=basis_name,
        n_functions=n_functions,
        argvals=argvals,
        dimension=dimension,
        add_intercept=True
    )

    # Generate data
    kl = KarhunenLoeve(basis_name=None, basis=basis, random_state=seed)
    kl.new(n_obs=n_obs, clusters_std='exponential')
    return kl


def simulate_data_multivariate(
    n_components,
    n_obs,
    n_points,
    basis_name,
    n_functions,
    seed
):
    """Simulation function for univariate functional data.

    Parameters
    ----------
    n_components: int
        Number of components to simulate.
    n_obs: int
        Number of observations to simulate.
    n_points: int
        Number of points used to simulate the data.
    basis_name: str
        Name of the basis to used.
    n_functions: int
        Number of basis functions to used.
    seed: int
        A seed for reproducibility.

    Returns
    -------
    KarhunenLoeve
        The simulated dataset.

    """
    # Define the argvals
    rng = np.random.RandomState(seed)
    start_points = rng.uniform(-1, 1, n_components)
    end_points = rng.uniform(-1, 1, n_components)
    argvals = [
        np.sort(np.linspace(begin, end, n_points))
        for begin, end in zip(start_points, end_points)
    ]

    # Define the basis
    basis = MultivariateBasis(
        simulation_type='split',
        n_components=n_components,
        name=basis_name,
        n_functions=n_functions,
        argvals=argvals,
        dimension=n_components * ['1D']
    )

    # Generate data
    kl = KarhunenLoeve(basis_name=None, basis=basis, random_state=seed)
    kl.new(n_obs=n_obs, clusters_std='exponential')
    return kl


def ufpca_covariance(data, n_components):
    """Perform an estimation of the eigencomponents using the diagonalization
    of the covariance operator.

    Parameters
    ----------
    data: DenseFunctionalData
        A dataset
    n_components: Union[int, float, None]
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.

    Returns
    -------
    Union[UFPCA, FCPTPA]
        An object representing the diagonalization of the covariance operator.

    """
    if data.n_dim == 1:
        ufpca = UFPCA(
            n_components=n_components, method='covariance', normalize=False
        )
        ufpca.fit(data)
    elif data.n_dim == 2:
        n_points = data.n_points
        mat_v = make_diff_op(2, n_points['input_dim_0'])
        mat_w = make_diff_op(2, n_points['input_dim_1'])

        ufpca = FCPTPA(n_components=n_components, normalize=True)
        ufpca.fit(
            data,
            penalty_matrices={
                'v': np.dot(mat_v.T, mat_v),
                'w': np.dot(mat_w.T, mat_w)
            },
            alpha_range={
                'v': (1e-4, 1e4),
                'w': (1e-4, 1e4)
            },
            tolerance=1e-4, max_iteration=15, adapt_tolerance=True
        )
    return ufpca


def mfpca_covariance(data, n_components):
    """Perform an estimation of the eigencomponents using the diagonalization
    of the covariance operator.

    Parameters
    ----------
    data: MultivariateFunctonalData
        A dataset
    n_components: Union[int, float, None]
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.

    Returns
    -------
    MFPCA
        An object representing the diagonalization of the covariance operator.

    """
    n_components = data.n_functional * [n_components]
    mfpca = MFPCA(
        n_components=n_components, method='covariance', normalize=True
    )
    mfpca.fit(data)
    return mfpca


def ufpca_inner_product(data, n_components):
    """Perform an estimation of the eigencomponents using the diagonalization
    of the inner product matrix.

    Parameters
    ----------
    data: DenseFunctionalData
        A dataset
    n_components: Union[int, float, None]
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.

    Returns
    -------
    UFPCA
        An object representing the diagonalization of the inner-product matrix.

    """
    ufpca = UFPCA(n_components=n_components, method='inner-product')
    ufpca.fit(data)
    return ufpca


def mfpca_inner_product(data, n_components):
    """Perform an estimation of the eigencomponents using the diagonalization
    of the inner product matrix.

    Parameters
    ----------
    data: MultivariateFunctionalData
        A dataset
    n_components: Union[int, float, None]
        Number of components to keep. If `n_components` is `None`, all
        components are kept, ``n_components == min(n_samples, n_features)``.
        If `n_components` is an integer, `n_components` are kept. If
        `0 < n_components < 1`, select the number of components such that the
        amount of variance that needs to be explained is greater than the
        percentage specified by `n_components`.

    Returns
    -------
    MFPCA
        An object representing the diagonalization of the inner-product matrix.

    """
    mfpca = MFPCA(n_components=n_components, method='inner-product')
    mfpca.fit(data)
    return mfpca


def compute_scores(data, fpca):
    """Compute the scores using numerical integration.

    Parameters
    ----------
    data: Union[DenseFunctionalData, MultivariateFunctionalData]
        A dataset.
    fpca: Union[UFPCA, FCPTPA, MFPCA]
        An UFPCA or FCPTPA or MFPCA object.

    Returns
    -------
    npt.NDArray[np.float64]
        An estimation of the projection of the scores using numerical
        integration.

    """
    if isinstance(fpca, FCPTPA):
        return fpca.transform(data)
    else:
        if fpca.method == 'inner-product':
            return fpca.transform(data, method='InnPro')
        else:
            return fpca.transform(data, method='NumInt')


def compute_reconstruction(fpca, scores):
    """Compute the reconstruction of the curves.

    Parameters
    ----------
    ufpca: Union[UFPCA, FCPTPA, MFPCA]
        An UFPCA or FCPTPAobject.
    scores: npt.NDArray[np.float64]
        The scores

    Returns
    -------
    Union[DenseFunctionalData, MultivariateFunctionalData]
        The reconstructed dataset.

    """
    return fpca.inverse_transform(scores)


def flip(
    data: DenseFunctionalData,
    data_reference: DenseFunctionalData
):
    """Flip data if they have opposite sign."""
    norm_pos = np.linalg.norm(data.values + data_reference.values)
    norm_neg = np.linalg.norm(data.values - data_reference.values)
    
    sign = -1 if norm_pos < norm_neg else 1
    return DenseFunctionalData(data.argvals, sign * data.values)


def flip_multi(
    data: MultivariateFunctionalData,
    data_reference: MultivariateFunctionalData
):
    """Flip data if they have opposite sign."""
    data_list = data.n_functional * [None]
    for idx in np.arange(data.n_functional):
        data_list[idx] = flip(data[idx], data_reference[idx])
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
