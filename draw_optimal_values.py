#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tools import *
import argparse
import envs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        '--gamma',
        type=float,
        required=True,
        help='')
    parser.add_argument(
        '--outfile',
        type=str,
        required=True,
        help='')
    args = vars(parser.parse_args())
    assert(0 <= args['gamma'] < 1)
    return args


def run(gamma, outfile):
    env = envs.GridWorld(gamma)
    vmin, vmax = env.get_value_range()
    nan_grid = np.ones(env.walls.shape, dtype=np.float32) * np.nan
    walls_grid = np.zeros(env.walls.shape, dtype=np.float32)
    walls_grid[env.walls] = np.nan
    zero_grid = np.zeros(env.walls.shape, dtype=np.float32)

    # bulid drawing
    fig, axmat = plt.subplots(env.card[0],
                              env.card[1],
                              figsize=(env.card[0], env.card[1]),
                              dpi=300)
    plot_index = 0
    for x in range(env.card[0]):
        for y in range(env.card[1]):
            ax = axmat[x, y]
            if np.isnan(walls_grid[x, y]):  # wall square (outer)
                plot_index += 1
                print('Plotting {} of 257'.format(plot_index),
                      end=('\r' if plot_index < 257 else '\n'))
                sns.heatmap(zero_grid,
                            xticklabels=False,
                            yticklabels=False,
                            cbar=False,
                            linewidth=0,
                            square=True,
                            ax=ax)
            else:  # value square (outer)

                # plot values
                grid = env.get_optimal_values(terminal_state=np.array([x, y]))
                plot_index += 1
                print('Plotting {} of 257'.format(plot_index),
                      end=('\r' if plot_index < 257 else '\n'))
                sns.heatmap(grid,  # value square (inner)
                            vmin=vmin,
                            vmax=vmax,
                            mask=np.isnan(grid),
                            center=0.5,
                            cmap=sns.color_palette("coolwarm",
                                                   np.prod(grid.shape) ** 3,
                                                   as_cmap=True),
                            xticklabels=False,
                            yticklabels=False,
                            cbar=False,
                            linewidth=0,
                            square=True,
                            ax=ax)

                # plot walls
                plot_index += 1
                print('Plotting {} of 257'.format(plot_index),
                      end=('\r' if plot_index < 257 else '\n'))
                sns.heatmap(zero_grid,
                            mask=np.invert(np.isnan(walls_grid)),
                            xticklabels=False,
                            yticklabels=False,
                            cbar=False,
                            linewidth=0,
                            square=True,
                            ax=ax)

                # plot goal
                nan_grid[x, y] = 0
                plot_index += 1
                print('Plotting {} of 257'.format(plot_index),
                      end=('\r' if plot_index < 257 else '\n'))
                sns.heatmap(nan_grid,
                            mask=np.isnan(nan_grid),
                            cmap=sns.diverging_palette(0, 0, s=100, l=100, n=1),
                            xticklabels=False,
                            yticklabels=False,
                            cbar=False,
                            linewidth=0,
                            square=True,
                            ax=ax)
                ax.scatter([y + 0.5], [x + 0.5], marker="x", color="black", s=7.5)
                nan_grid[x, y] = np.nan

    # save drawing
    fig.savefig(outfile)


def main(args):
    run(args['gamma'], args['outfile'])


if __name__ == '__main__':
    main(parse_args())
