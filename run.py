#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tools import *
import argparse
import datetime
import envs
import json
import numpy as np
import time
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        'mode',
        choices=['policy_iteration', 'new'],
        help='')
    parser.add_argument(
        '--gamma',
        type=float,
        required=True,
        help='')
    parser.add_argument(
        '--epsilon',
        type=float,
        required=False,
        help='')
    parser.add_argument(
        '--num-iter',
        type=int,
        required=True,
        help='')
    parser.add_argument(
        '--use-wandb',
        action='store',
        const=True,
        default=False,
        nargs='?',
        help='')
    parser.add_argument(
        '--record-outfile',
        type=str,
        required=False,
        help='')
    args = vars(parser.parse_args())
    assert(0 <= args['gamma'] < 1)
    assert(0 < args['epsilon'])
    assert(0 < args['num_iter'])
    if args['use_wandb'] == '':
        args['use_wandb'] = True
    return args


def run(env,
        mode,
        gamma,
        epsilon,
        num_iter,
        use_wandb=False):
    record = list()

    if mode == 'policy_iteration':
        action_map = np.zeros(env.card, dtype=np.uint8)

        def policy(action, state):
            return 1 if action_map[state[0], state[1]] == action else 0
    else:
        weights = np.ones((env.card.tolist() + [env.num_actions])) * 0.25
        for x in range(env.card[0]):
            for y in range(env.card[1]):
                if env.is_wall((x, y)):
                    weights[x, y, :].fill(0)

        def policy(action, state):
            return weights[state[0], state[1], action]

    optimal_values = env.get_optimal_values()

    for i in range(num_iter):
        start_time = time.time()

        # update policy one step
        state_values = env.get_state_values(policy, epsilon)
        action_values = env.get_action_values(policy, epsilon)
        for x in range(env.card[0]):
            for y in range(env.card[1]):
                if env.is_wall((x, y)) or env.is_terminal((x, y)):
                    continue
                for action in range(env.num_actions):
                    if mode == 'policy_iteration':
                        if action_values[x, y, action] > action_values[x, y, action_map[x, y]]:
                            action_map[x, y] = action
                    else:
                        weights[x, y, action] *= action_values[x, y, action] / state_values[x, y]

        # record learning progress
        value_error = list()
        for x in range(env.card[0]):
            for y in range(env.card[1]):
                if env.is_wall((x, y)) or env.is_terminal((x, y)):
                    continue
                value_error.append(optimal_values[x, y] - state_values[x, y])
        entry = {
            'Initial State Absolute Value Error': float(abs(optimal_values[1, 1] - state_values[1, 1])),
            'RMSVE': float(np.sqrt(np.mean(np.square(value_error))))
        }
        record.append(entry)
        if use_wandb:
            wandb.log(entry)

        # print something to show progress
        print('Iteration: {0} (took {1:.4f} seconds)'.format(i + 1, time.time() - start_time))

    return record


def main(args):
    if args['use_wandb']:
        wandb.login()
        if not isinstance(args['use_wandb'], bool):
            assert(isinstance(args['use_wandb'], str))
            wandb.init(project='reward-weighted-regression',
                       config=args,
                       name=args['use_wandb'])
        else:
            wandb.init(project='reward-weighted-regression',
                       config=args)

    env = envs.GridWorld(args['gamma'])

    # record start time
    start_time = datetime.datetime.now().isoformat()

    record = run(
        env,
        args['mode'],
        args['gamma'],
        args['epsilon'],
        args['num_iter'],
        use_wandb=args['use_wandb'])

    # record the end time
    end_time = datetime.datetime.now().isoformat()

    # if requested then save the results
    if args['record_outfile'] is not None:
        results = dict()
        results['start_time'] = start_time
        results['end_time'] = end_time
        results['args'] = args
        results['record'] = list_of_dicts_to_dict_of_lists(record)
        with open(args['record_outfile'], 'w') as outfile:
            json.dump(results, outfile)


if __name__ == '__main__':
    main(parse_args())