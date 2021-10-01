#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import envs
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("outfile", type=str, help="")
    args = vars(parser.parse_args())
    return args


def run(outfile):
    fig, axarr = plt.subplots(2, 1, figsize=(2, 4), dpi=300)
    axarr = np.append(axarr, axarr[1].twinx())
    sns.set_style("ticks")
    sns.set_context("notebook")

    env = envs.GridWorld()
    vmin = env.min_value
    vmax = env.max_value
    walls_grid = np.zeros(env.walls.shape, dtype=np.float32)
    walls_grid[env.walls] = np.nan
    zero_grid = np.zeros(env.walls.shape, dtype=np.float32)

    # plot values
    grid = env.get_optimal_values()
    sns.heatmap(
        grid,  # value square (inner)
        vmin=vmin,
        vmax=vmax,
        mask=np.isnan(grid),
        center=0.5,
        cmap=sns.color_palette("coolwarm", np.prod(grid.shape) ** 3, as_cmap=True),
        cbar_kws={"location": "left", "label": "Value", "ticks": [0.0, 0.5, 1.0]},
        xticklabels=False,
        yticklabels=False,
        linewidth=0,
        square=True,
        ax=axarr[0],
    )

    # plot walls
    sns.heatmap(
        zero_grid,
        mask=np.invert(np.isnan(walls_grid)),
        xticklabels=False,
        yticklabels=False,
        cbar=False,
        linewidth=0,
        square=True,
        ax=axarr[0],
    )

    # load data
    raw_data = list()
    for i in range(200):
        with open("{}.json".format(i), "r") as infile:
            raw_data.append(json.load(infile))
    data = {
        k1: {k2: list() for k2 in ["RMSVE", "Return"]}
        for k1 in ["reward_weighted_regression", "policy_iteration"]
    }
    for entry in raw_data:
        for k2 in ["RMSVE", "Return"]:
            data[entry["args"]["mode"]][k2].append(entry["record"][k2])
    for k1 in data.keys():
        for k2 in data[k1].keys():
            data[k1][k2] = np.array(data[k1][k2])
    colors = sns.color_palette("colorblind", 1)

    # plot RMSVE
    k1 = "reward_weighted_regression"
    k2 = "RMSVE"
    x = np.arange(41)
    y = np.mean(data[k1][k2], axis=0)
    yerr = np.std(data[k1][k2], axis=0)
    axarr[1].plot(x, y, color=colors[0])
    axarr[1].fill_between(x, y - yerr, y + yerr, color=colors[0], alpha=0.3)
    axarr[1].set_xlabel("Iteration", labelpad=5)
    axarr[1].set_ylabel("RMSVE (solid line)", labelpad=7.5)
    axarr[1].set_xticks([0, 20, 40])
    axarr[1].set_yticks([0.0, 0.25, 0.5])
    axarr[1].set_xlim(0, 41)
    axarr[1].set_ylim(-0.05, 0.55)

    # plot initial state value
    k1 = "reward_weighted_regression"
    k2 = "Return"
    x = np.arange(41)
    y = np.mean(data[k1][k2], axis=0)
    yerr = np.std(data[k1][k2], axis=0)
    axarr[2].plot(x, y, linestyle="--", color=colors[0])
    axarr[2].fill_between(x, y - yerr, y + yerr, color=colors[0], alpha=0.3)
    axarr[2].set_ylabel("Return (dashed line)", labelpad=7.5)
    axarr[2].set_yticks([0.0, 0.1, 0.2])
    axarr[2].set_xlim(0, 41)
    axarr[2].set_ylim(-0.02, 0.23)

    # clean up and save plots
    fig.subplots_adjust(bottom=0.25, hspace=0.4)
    fig.savefig(outfile, bbox_inches="tight")


def main(args):
    run(args["outfile"])


if __name__ == "__main__":
    main(parse_args())
