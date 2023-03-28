################################################################################
# Plot functions for the simulations
# S. Golovkine - 17/01/2023
################################################################################

# Load packages
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

# Config
plt.style.use('./code/stylefiles/plot.mplstyle')

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
    data['ratio_time'] = data['time_inn'] / data['time_cov']
    data['ratio_mise'] = data['mise_inn'] / data['mise_cov']

    # Select data time
    data_time = data.loc[:, ['P', 'N', 'M', 'ratio_time']].\
        sort_values(['P', 'N', 'M']).\
        replace({'P': {2: '$P = 2$', 10: '$P = 10$', 20: '$P = 20$', 50: '$P = 50$'}})

    # Select data MISE
    data_mise = data.loc[:, ['P', 'N', 'M', 'ratio_mise']].\
        sort_values(['P', 'N', 'M']).\
        replace({'P': {2: '$P = 2$', 10: '$P = 10$', 20: '$P = 20$', 50: '$P = 50$'}})
    return data_time, data_mise


def plot_mise(results):
    """Plot the results of the simulation in term of MISE."""
    gg = sns.catplot(
        data=results,
        x="ratio_mise", y="M", col="N", hue="P",
        kind="box", orient="h",
        sharex=False, margin_titles=True,
        col_wrap=2, height=3, aspect=3
    )
    #gg.set(xlabel="Ratio of MISE", ylabel="")
    gg.set_titles(row_template="N =$ {row_name}", size=30)
    gg.set_yticklabels(["$M = 25$", "$M = 50$", "$M = 75$", "$M = 100$"], size=20)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=0.8, ls='--')
        ax.set_title(f"${ax.get_title()}$")
        ax.set_xlabel("Ratio of MISE", size=25)
        ax.set_ylabel("")
    gg.fig.tight_layout()
    sns.move_legend(
        gg, "lower center",
        bbox_to_anchor=(.5, 0), title=None, frameon=True, ncol=3, columnspacing=0.6
    )
    return gg


def plot_computation_time(results):
    """Plot the results of the simulation in term of computation time."""
    gg = sns.catplot(
        data=results,
        x="ratio_time", y="M", col="N", hue="P",
        kind="violin", orient="h",
        sharex=False, margin_titles=False,
        col_wrap=2, height=3, aspect=3
    )
    #gg.set(xlabel="Ratio of computation time", ylabel="")
    gg.set_titles(row_template="N = {row_name}", size=30)
    gg.set_yticklabels(["$M = 25$", "$M = 50$", "$M = 75$", "$M = 100$"], size=20)
    for ax in gg.axes.flat:
        ax.axvline(x=1, color='r', lw=0.8, ls='--')
        ax.set_title(f"${ax.get_title()}$")
        ax.set_xlabel("Ratio of MISE", size=25)
        ax.set_ylabel("")
    gg.fig.tight_layout()
    sns.move_legend(
        gg, "lower center",
        bbox_to_anchor=(.5, 0), title=None, frameon=True, ncol=3, columnspacing=0.6
    )
    return gg


# Run
if __name__ == "__main__":
    # Load data
    PATH = f"{args.src_results}"

    results_time, results_mise = load_results(PATH)

    path_split = re.split("/", PATH)
    
    # Create directory
    if not os.path.exists(f"{args.out_folder}/{path_split[1]}"):
        os.makedirs(f"{args.out_folder}/{path_split[1]}")

    # Plot computation time
    plot_computation_time(results_time)
    # plt.show()
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/computation_time.{args.format}',
        format=args.format
    )
    plt.close()

    # Plot MISE
    plot_mise(results_mise)
    # plt.show()
    plt.savefig(
        f'{args.out_folder}/{path_split[1]}/mise.{args.format}',
        format=args.format
    )
    plt.close()
