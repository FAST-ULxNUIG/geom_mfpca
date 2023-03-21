################################################################################
# Utility functions for the simulations of univariate functional data
# S. Golovkine - 17/01/2023
################################################################################

# Load packages
import numpy as np

from FDApy.preprocessing.dim_reduction.fpca import UFPCA, MFPCA
from FDApy.preprocessing.dim_reduction.fcp_tpa import FCPTPA
from FDApy.representation.basis import Basis
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve

# Variables

# Functions
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
        Number of basis functions to used. For 2D-data, the number of functions
        used is n_functions * n_functions.
    dimension: str
        Dimension of the data (1D or 2D).
    seed: int
        A seed for reproducibility.

    Returns
    -------
    DenseFunctionalData
        The simulated dataset.

    """
    argvals = {'input_dim_0': np.linspace(0, 1, n_points)}
    kl = KarhunenLoeve(
        basis_name=basis_name,
        n_functions=n_functions,
        dimension=dimension,
        argvals=argvals,
        random_state=seed
    )
    kl.new(n_obs=n_obs, clusters_std='exponential')
    return kl.data


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
    DenseFunctionalData
        The simulated dataset.

    """
    argvals = np.linspace(0, 10, num=n_components * (n_points + 1))

    basis = Basis(
        name=basis_name,
        n_functions=n_functions,
        dimension='1D',
        argvals={'input_dim_0': argvals}
    )
    basis = basis[1:]

    # Select the splitting points
    cut_points_idx = np.arange(0, len(argvals), n_points)

    # Create Multivariate Basis
    basis_multi = n_components * [None]
    for i in np.arange(n_components):
        start_idx = cut_points_idx[i]
        end_idx = cut_points_idx[i + 1]

        new_argvals = argvals[start_idx:end_idx]
        deno = (np.max(new_argvals) - np.min(new_argvals))
        new_argvals = (new_argvals - np.min(new_argvals)) / deno

        sign = np.random.choice([-1, 1])
        new_values = sign * basis.values[:, start_idx:end_idx]
        data = DenseFunctionalData({'input_dim_0': new_argvals}, new_values)
        data.dimension = '1D'
        basis_multi[i] = data

    kl = KarhunenLoeve(
        basis_name=None, basis=basis_multi, random_state=seed
    )
    kl.new(n_obs=n_obs, clusters_std='exponential')
    return kl.data


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
        mat_v = np.diff(np.identity(n_points['input_dim_0']))
        mat_w = np.diff(np.identity(n_points['input_dim_1']))
        ufpca = FCPTPA(n_components=n_components)
        ufpca.fit(
            data,
            penalty_matrices={
                'v': np.dot(mat_v, mat_v.T),
                'w': np.dot(mat_w, mat_w.T)
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
        n_components=n_components, method='covariance', normalize=False
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


def MISE_1D(data_true, data_recons):
    """Compute MISE between two univariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data_true: DenseFunctionalData
        Dataset of functional data.
    data_recons: DenseFunctionalData
        Data of functionalData.

    Returns
    -------
    float
        The mean integrated squared error between the two datasets.

    """
    values_true = data_true.values
    values_recons = data_recons.values
    diff_squared = np.power(values_true - values_recons, 2)
    int_along_x = np.trapz(y=diff_squared, x=data_true.argvals['input_dim_0'])
    return np.mean(int_along_x)


def MISE_2D(data_true, data_recons):
    """Compute 2D-MISE between two functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data_true: DenseFunctionalData
        Dataset of functional data.
    data_recons: DenseFunctionalData
        Data of functionalData.

    Returns
    -------
    float
        The mean integrated squared error between the two datasets.

    """
    values_true = data_true.values
    values_recons = data_recons.values
    diff_squared = np.power(values_true - values_recons, 2)
    int_along_y = np.trapz(y=diff_squared, x=data_true.argvals['input_dim_1'])
    int_along_x = np.trapz(y=int_along_y, x=data_true.argvals['input_dim_0'])
    return np.mean(int_along_x)


def MISE(data_true, data_recons):
    """Compute MISE between two multivariate functional datasets.

    The two datasets must have the same number of observations and the same
    argvals.

    Parameters
    ----------
    data_true: MultivariateFunctionalData
        Dataset of functional data.
    data_recons: MultivariateFunctionalData
        Data of functionalData.

    Returns
    -------
    float
        The mean integrated squared error between the two datasets.

    """
    return np.sum([MISE_1D(d, d_f) for d, d_f in zip(data_true, data_recons)])
