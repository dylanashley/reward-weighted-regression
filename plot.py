#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import envs
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        'outfile',
        type=str,
        help='')
    args = vars(parser.parse_args())
    return args


def run(outfile):
    fig, axarr = plt.subplots(1, 3, figsize=(10, 2), dpi=300)
    sns.set_style('ticks')
    sns.set_context('notebook')

    env = envs.GridWorld()
    vmin = env.min_value
    vmax = env.max_value
    nan_grid = np.ones(env.walls.shape, dtype=np.float32) * np.nan
    walls_grid = np.zeros(env.walls.shape, dtype=np.float32)
    walls_grid[env.walls] = np.nan
    zero_grid = np.zeros(env.walls.shape, dtype=np.float32)

    # plot values
    grid = env.get_optimal_values()
    sns.heatmap(grid,  # value square (inner)
                vmin=vmin,
                vmax=vmax,
                mask=np.isnan(grid),
                center=0.5,
                cmap=sns.color_palette('coolwarm', np.prod(grid.shape) ** 3, as_cmap=True),
                cbar_kws={'location': 'left', 'label': 'Value', 'ticks': [0.0, 0.5, 1.0]},
                xticklabels=False,
                yticklabels=False,
                linewidth=0,
                square=True,
                ax=axarr[0])

    # plot walls
    sns.heatmap(zero_grid,
                mask=np.invert(np.isnan(walls_grid)),
                xticklabels=False,
                yticklabels=False,
                cbar=False,
                linewidth=0,
                square=True,
                ax=axarr[0])

    # plot goal
    #nan_grid[9, 9] = 0
    #sns.heatmap(nan_grid,
    #            mask=np.isnan(nan_grid),
    #            cmap=sns.diverging_palette(0, 0, s=100, l=100, n=1),
    #            xticklabels=False,
    #            yticklabels=False,
    #            cbar=False,
    #            linewidth=0,
    #            square=True,
    #            ax=axarr[0])
    #axarr[0].scatter([9.5], [9.5], marker="x", color="black", s=30)
    #nan_grid[9, 9] = np.nan

    # load data
    raw_data = list()
    for i in range(20): #200
        with open('{}.json'.format(i), 'r') as infile:
            raw_data.append(json.load(infile))
    data = {k1: {k2: list() for k2 in ['RMSVE', 'Return']}
            for k1 in ['reward_weighted_regression', 'policy_iteration']}
    for entry in raw_data:
        for k2 in ['RMSVE', 'Return']:
            data[entry['args']['mode']][k2].append(entry['record'][k2])
    for k1 in data.keys():
        for k2 in data[k1].keys():
            data[k1][k2] = np.array(data[k1][k2])
    colors = sns.color_palette('colorblind', 2)

    # plot RMSVE
    k2 = 'RMSVE'
    for i, k1 in enumerate(['reward_weighted_regression', 'policy_iteration']):
        label = {'reward_weighted_regression': 'RWR',
                 'policy_iteration': 'PI'}[k1]
        x = np.arange(41)
        y = np.mean(data[k1][k2], axis=0)
        yerr = np.std(data[k1][k2], axis=0)
        axarr[1].plot(
            x,
            y,
            color=colors[i],
            label=label)
        axarr[1].fill_between(
            x,
            y - yerr,
            y + yerr,
            color=colors[i],
            alpha=0.3)
    axarr[1].set_xlabel('Iteration', labelpad=5)
    axarr[1].set_ylabel('RMSVE', labelpad=5)
    axarr[1].set_xticks([0, 20, 40])
    axarr[1].set_yticks([0.0, 0.25, 0.5])
    axarr[1].set_xlim(0, 41)
    axarr[1].set_ylim(- 0.05, 0.55)
    axarr[1].legend(loc='best', frameon=False)

    # plot initial state value
    k2 = 'Return'
    for i, k1 in enumerate(['reward_weighted_regression', 'policy_iteration']):
        x = np.arange(41)
        y = np.mean(data[k1][k2], axis=0)
        yerr = np.std(data[k1][k2], axis=0)
        axarr[2].plot(
            x,
            y,
            color=colors[i])
        axarr[2].fill_between(
            x,
            y - yerr,
            y + yerr,
            color=colors[i],
            alpha=0.3)
    axarr[2].set_xlabel('Iteration', labelpad=5)
    axarr[2].set_ylabel('Return', labelpad=5)
    axarr[2].set_xticks([0, 20, 40])
    axarr[2].set_yticks([0.0, 0.1, 0.2])
    axarr[2].set_xlim(0, 41)
    axarr[2].set_ylim(- 0.02, 0.24)

    # clean up and save plots
    fig.subplots_adjust(bottom=0.25, wspace=0.4)
    fig.savefig(outfile, bbox_inches='tight')


def main(args):
    run(args['outfile'])


if __name__ == '__main__':
    main(parse_args())
