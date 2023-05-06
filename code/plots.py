################################################################################
# Plot functions for the simulations
# S. Golovkine - 17/01/2023
################################################################################

# Load packages
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

# Config
plt.style.use('./code/stylefiles/plot.mplstyle')
mpl.rcParams['xtick.labelsize'] = 10

# Argument parser for the file
parser = argparse.ArgumentParser(
    description="Plots of the results.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("src_results", help="Path of the result files")
parser.add_argument("out_folder", help="Output folder to store the results")
parser.add_argument(
    "-f", "--format", default='eps', help="Format of the output"
)
args = parser.parse_args()

# Functions
def load_results(path):
    """Load the results.
    
    Parameters
    ----------
    path: str
        Path of the result files.

    Returns
    -------
    pd.Dataframes
        The results

    """
    # List the files
    results_files = os.listdir(path)

    # Load the data
    list_data = len(results_files) * [None]
    for idx, file in enumerate(results_files):
        if file[0] == '.':
            continue
        file_split = re.split("_", file)
        N = int(file_split[1])
        M = int(file_split[2])
        P = int(file_split[3])

        data = pd.read_json(f'{PATH}/{file}')
        data.insert(0, "P", P)
        data.insert(0, "M", M)
        data.insert(0, "N", N)
        list_data[idx] = data

    # Concatenate the dataframes
    data = pd.concat(list_data)

    # Compute ratio inner-product / covariance
    data['ratiotime'] = data['time_inn'] / data['time_cov']
    data['ratiomise'] = data['mise_inn'] / data['mise_cov']

    # Compute ratio inner-product / covariance logAE
    data_logAE = data[['N', 'M', 'P', 'logAE_cov', 'logAE_inn']].\
        explode(['logAE_cov', 'logAE_inn'])
    data_logAE['K'] = np.tile(np.arange(5) + 1, len(data))
    data_logAE['ratiologAE'] = data_logAE['logAE_inn'] / data_logAE['logAE_cov']
    data_logAE['ratiologAE'] = data_logAE['ratiologAE'].astype('float')

    # Compute ratio inner-product / covariance ISE
    data_ise = data[['N', 'M', 'P', 'ise_cov', 'ise_inn']].\
        explode(['ise_cov', 'ise_inn'])
    data_ise['K'] = np.tile(np.arange(5) + 1, len(data))
    data_ise['ratioise'] = data_ise['ise_inn'] / data_ise['ise_cov']

    # Select data time
    data_time = data.loc[:, ['P', 'N', 'M', 'ratiotime']].\
        sort_values(['P', 'N', 'M']).\
            replace({'P': {2: '2', 10: '10', 20: '20', 50: '50'}}).\
            replace({'M': {25: '25', 50: '50', 75: '75', 100: '100'}})

    # Select data MISE
    data_mise = data.loc[:, ['P', 'N', 'M', 'ratiomise']].\
        sort_values(['P', 'N', 'M']).\
        replace({'P': {2: '2', 10: '10', 20: '20', 50: '50'}}).\
        replace({'M': {25: '25', 50: '50', 75: '75', 100: '100'}})

    # Select data logAE
    data_logAE = data_logAE.loc[:, ['P', 'N', 'M', 'K', 'ratiologAE']].\
        sort_values(['P', 'N', 'M', 'K']).\
        replace({'P': {2: '2', 10: '10', 20: '20', 50: '50'}}).\
        replace({'M': {25: '25', 50: '50', 75: '75', 100: '100'}}).\
        replace({'K': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}})

    # Select data logAE
    data_ise = data_ise.loc[:, ['P', 'N', 'M', 'K', 'ratioise']].\
        sort_values(['P', 'N', 'M', 'K']).\
        replace({'P': {2: '2', 10: '10', 20: '20', 50: '50'}}).\
        replace({'M': {25: '25', 50: '50', 75: '75', 100: '100'}}).\
        replace({'K': {1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}})

    return data_time, data_mise, data_logAE, data_ise


def plot_mise(results):
    """Plot the results of the simulation in term of MISE."""
    gg = sns.catplot(
        data=results,
        x="ratiomise", y="P", col="N", row="M",
        kind="box",
        flierprops=dict(marker="+", markerfacecolor="gray", markersize=1),
        height=3,
        aspect=1
    )
    gg.set_titles(template="$N = {col_name} \:|\: M = {row_name}$", size=12)
    gg.set(xlim=(0, 1.5))
    gg.set_xlabels("MISE($\mathcal{X}$, $\widehat{\mathcal{X}}$) / MISE($\mathcal{X}$, $\widetilde{\mathcal{X}}$)", fontsize=10)
    gg.set_ylabels("")
    gg.set_yticklabels(["$P = 2$", "$P = 10$", "$P = 20$", "$P = 50$"], size=10)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=1, ls='--')
        ax.set_xscale("linear")
    gg.fig.tight_layout()
    return gg


def plot_computation_time(results):
    """Plot the results of the simulation in term of computation time."""
    gg = sns.catplot(
        data=results,
        x="ratiotime", y="P", col="N", row="M",
        kind="violin",
        height=3,
        aspect=1
    )
    gg.set_titles(template="$N = {col_name} \:|\: M = {row_name}$", size=12)
    gg.set(xlim=(10e-3, 30))
    gg.set_xlabels("Ratio of computation time", fontsize=10)
    gg.set_ylabels("")
    gg.set_yticklabels(["$P = 2$", "$P = 10$", "$P = 20$", "$P = 50$"], size=10)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=1, ls='--')
        ax.set_xscale("log")
    gg.fig.tight_layout()
    return gg


def plot_ise(results):
    """Plot the results of the simulation in term of ISE."""
    gg = sns.catplot(
        data=results.replace({'M': {'25': '$M = 25$', '50': '$M = 50$', '75': '$M = 75$', '100': '$M = 100$'}}),
        x="ratioise", y="P", col="K", row="N", hue="M",
        kind="box", flierprops=dict(marker="+", markerfacecolor="gray", markersize=1),
        height=3,
        aspect=1
    )
    gg.set_titles(template="$\phi_{col_name} \:|\: N = {row_name}$", size=12)
    gg.set(xlim=(0.8, 1.2))
    gg.set_xlabels("ISE($\phi_k$, $\widehat{\phi}_k$) / ISE($\phi_k$, $\widetilde{\phi}_k$)", fontsize=10)
    gg.set_ylabels("")
    gg.set_yticklabels(["$P = 2$", "$P = 10$", "$P = 20$", "$P = 50$"], size=10)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=1, ls='--')
        ax.set_xscale("linear")
    gg.fig.tight_layout()
    sns.move_legend(
        gg, 'lower center',
        bbox_to_anchor=(.5, -0.05),
        title='',
        frameon=True,
        ncol=4, columnspacing=0.5, fontsize=16
    )
    return gg


def plot_logAE(results):
    """Plot the results of the simulation in term of logAE."""
    gg = sns.catplot(
        data=results.replace({'M': {'25': '$M = 25$', '50': '$M = 50$', '75': '$M = 75$', '100': '$M = 100$'}}),
        x="ratiologAE", y="P", col="K", row="N", hue="M",
        kind="box", flierprops=dict(marker="+", markerfacecolor="gray", markersize=1),
        height=3,
        aspect=1
    )
    gg.set_titles(template="$\lambda_{col_name} \:|\: N = {row_name}$", size=12)
    gg.set(xlim=(0.5, 1.5))
    gg.set_xlabels("$\log-$AE($\lambda_k$, $\widehat{\lambda}_k$) / $\log-$AE($\lambda_k$, $\widetilde{\lambda}_k$)", fontsize=10)
    gg.set_ylabels("")
    gg.set_yticklabels(["$P = 2$", "$P = 10$", "$P = 20$", "$P = 50$"], size=10)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=1, ls='--')
        ax.set_xscale("linear")
    gg.fig.tight_layout()
    sns.move_legend(
        gg, 'lower center',
        bbox_to_anchor=(.5, -0.05),
        title='',
        frameon=True,
        ncol=4, columnspacing=0.5, fontsize=16
    )
    return gg

# Run
if __name__ == "__main__":
    # Load data
    PATH = f"{args.src_results}"

    results_time, results_mise, results_logAE, results_ise = load_results(PATH)

    path_split = re.split("/", PATH)

    # Create directory
    if not os.path.exists(f"{args.out_folder}/{path_split[1]}"):
        os.makedirs(f"{args.out_folder}/{path_split[1]}")
        os.makedirs(f"{args.out_folder}/{path_split[1]}/computation_time")
        os.makedirs(f"{args.out_folder}/{path_split[1]}/mise")
        os.makedirs(f"{args.out_folder}/{path_split[1]}/logAE")
        os.makedirs(f"{args.out_folder}/{path_split[1]}/ise")

    # Plot computation time
    plot_computation_time(results_time)
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/computation_time/'
        f'computation_time.{args.format}',
        format=args.format,
        bbox_inches='tight'
    )
    plt.close()

    # Plot MISE
    plot_mise(results_mise)
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/mise/'
        f'mise.{args.format}',
        format=args.format,
        bbox_inches='tight'
    )
    plt.close()

    # Plot log-AE
    gg = plot_logAE(results_logAE)
    print(gg)
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/logAE/'
        f'logAE.{args.format}',
        format=args.format,
        bbox_inches='tight'
    )
    plt.close()

    # Plot ISE
    plot_ise(results_ise)
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/ise/'
        f'ise.{args.format}',
        format=args.format,
        bbox_inches='tight'
    )
    plt.close()
