# -*- coding: utf-8 -*-

from queue import SimpleQueue
import numpy as np


class GridWorld:

    def __init__(self, gamma):
        self.gamma = gamma
        self.card = np.array([11, 11])
        self.num_actions = 4
        self.terminal_state = np.array([9, 9])
        self.walls = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=bool)
        self.reset()

    def get_state(self):
        return np.copy(self.state)

    def set_state(self, state):
        self.state = np.copy(state)

    def get_next_state(self, action, state=None):
        if state is None:
            state = self.get_state()
        else:
            state = np.copy(state)
        if action == 0:
            if not self.is_wall((state[0], state[1] - 1)):
                state[1] -= 1
        elif action == 1:
            if not self.is_wall((state[0] + 1, state[1])):
                state[0] += 1
        elif action == 2:
            if not self.is_wall((state[0], state[1] + 1)):
                state[1] += 1
        else:
            assert(action == 3)
            if not self.is_wall((state[0] - 1, state[1])):
                state[0] -= 1
        return state

    def is_terminal(self, state=None):
        if state is None:
            state = self.get_state()
        return (state[0] == self.terminal_state[0]) and \
               (state[1] == self.terminal_state[1])

    def is_wall(self, state):
        return self.walls[state[0], state[1]]

    def get_reward(self, state=None):
        if state is None:
            state = self.get_state()
        return 1 if self.is_terminal(state) else 0

    def reset(self):
        self.set_state(np.array([1, 1]))
        return self.get_state()

    def step(self, action):
        assert(not self.is_terminal())
        self.set_state(self.get_next_state(action))
        return self.get_state(), self.get_reward(), self.is_terminal()

    def get_state_values(self, policy, epsilon):
        rv = np.zeros(self.card, dtype=np.float32)
        delta = float('inf')
        while delta > epsilon:
            delta = 0
            for x in range(self.card[0]):
                for y in range(self.card[1]):
                    state = np.array([x, y])
                    if self.is_wall(state) or self.is_terminal(state):
                        continue
                    old_state_value = rv[x, y]
                    rv[x, y] = 0
                    for action in range(self.num_actions):
                        next_state = self.get_next_state(action, state)
                        reward = self.get_reward(next_state)
                        next_state_value = rv[next_state[0], next_state[1]]
                        rv[x, y] += policy(action, state) * \
                            (reward + self.gamma * next_state_value)
                    delta = max(delta, abs(old_state_value - rv[x, y]))
        return rv

    def get_optimal_values(self):

        def get_neighbours(state):
            rv = list()
            if not self.is_wall((state[0], state[1] - 1)):
                rv.append((state[0], state[1] - 1))
            if not self.is_wall((state[0] + 1, state[1])):
                rv.append((state[0] + 1, state[1]))
            if not self.is_wall((state[0], state[1] + 1)):
                rv.append((state[0], state[1] + 1))
            if not self.is_wall((state[0] - 1, state[1])):
                rv.append((state[0] - 1, state[1]))
            return rv

        rv = np.zeros(self.card, dtype=np.float32)
        queue = SimpleQueue()
        queue.put((self.terminal_state[0], self.terminal_state[1]))
        while not queue.empty():
            x1, y1 = queue.get()
            reward = self.get_reward((x1, y1))
            neighbours = get_neighbours((x1, y1))
            for x2, y2 in neighbours:
                if self.is_terminal((x2, y2)):
                    continue
                if rv[x2, y2] < reward + self.gamma * rv[x1, y1]:
                    rv[x2, y2] = reward + self.gamma * rv[x1, y1]
                    queue.put((x2, y2))
        return rv

    def get_action_values(self, policy, epsilon):
        state_values = self.get_state_values(policy, epsilon)
        rv = np.zeros(state_values.shape + (self.num_actions,), dtype=np.float32)
        for x in range(self.card[0]):
            for y in range(self.card[1]):
                state = np.array([x, y])
                if self.is_wall(state):
                    continue
                for action in range(self.num_actions):
                    next_state = self.get_next_state(action, state)
                    reward = self.get_reward(next_state)
                    rv[state[0], state[1], action] = reward + self.gamma * \
                        state_values[next_state[0], next_state[1]]
        return rv

    def print_grid(self, grid):
        to_print = ''
        if isinstance(grid[0, 0], np.uint8):
            for row in np.transpose(grid):
                to_print += ' '.join([str(v) for v in row]) + '\n'
        elif isinstance(grid[0, 0], np.float32):
            for row in np.transpose(grid):
                to_print += ' '.join(['{0:.2f}'.format(v) for v in row]) + '\n'
        else:
            raise TypeError('unsupported type {}'.format(type(grid[0, 0])))
        print(to_print)

    def show(self, state=None):
        if state is None:
            state = self.get_state()
        sketch = self.walls.astype(np.uint8)
        sketch[state[0], state[1]] = 2
        sketch[self.terminal_state[0], self.terminal_state[1]] = 3
        self.print_grid(sketch)
