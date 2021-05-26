# -*- coding: utf-8 -*-

import numpy as np


class GridWorld:

    def __init__(self, gamma=0.9, initial_state=(1, 1), terminal_state=(9, 9)):
        assert(0 <= gamma < 1)
        assert((initial_state[0] != terminal_state[0]) or (initial_state[1] != terminal_state[1]))
        self.gamma = gamma
        self.initial_state = np.array(initial_state, dtype=np.uint8)
        self.terminal_state = np.array(terminal_state, dtype=np.uint8)
        self.card = np.array([11, 11])
        self.min_value = 0
        self.max_value = 1
        self.num_actions = 4
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
        assert(self._is_valid(self.initial_state))
        assert(self._is_valid(self.terminal_state))
        self.reset()

    def _is_wall(self, state):
        return self.walls[state[0], state[1]]

    def _is_valid(self, state):
        if len(state) != 2:
            return False
        if not ((0 < state[0] < 10) and (0 < state[1] < 10)):
            return False
        if self._is_wall(state):
            return False
        return True

    def _is_terminal(self, state):
        return (state[0] == self.terminal_state[0]) and \
               (state[1] == self.terminal_state[1])

    def set_state(self, state):
        assert(self._is_valid(state))
        self.state = np.copy(state)

    def get_state(self):
        return np.copy(self.state)

    def get_next_state(self, action, state=None):
        if state is None:
            state = self.get_state()
        else:
            state = np.copy(state)
            assert(self._is_valid(state))
        if not self._is_terminal(state):
            if action == 0:
                if not self._is_wall((state[0], state[1] - 1)):
                    state[1] -= 1
            elif action == 1:
                if not self._is_wall((state[0] + 1, state[1])):
                    state[0] += 1
            elif action == 2:
                if not self._is_wall((state[0], state[1] + 1)):
                    state[1] += 1
            else:
                assert(action == 3)
                if not self._is_wall((state[0] - 1, state[1])):
                    state[0] -= 1
        return state

    def get_reward(self, old_state, new_state):
        assert(self._is_valid(old_state))
        assert(self._is_valid(new_state))
        return 1 if self._is_terminal(new_state) and not self._is_terminal(old_state) else 0

    def reset(self):
        self.set_state(self.initial_state)
        return self.get_state()

    def step(self, action):
        assert(not self._is_terminal(self.get_state()))
        old_state = self.get_state()
        new_state = self.get_next_state(action)
        self.set_state(new_state)
        reward = self.get_reward(old_state, new_state)
        done = self._is_terminal(new_state)
        return new_state, reward, done

    def render(self, show=True):
        state = self.get_state()
        sketch = self.walls.astype(np.uint8)
        sketch[state[0], state[1]] = 2
        sketch[self.terminal_state[0], self.terminal_state[1]] = 3
        sketch = self.pretty_print(sketch)
        if show:
            print(sketch)
        return sketch

    def get_state_values(self, policy, epsilon=1e-6):
        assert(0 < epsilon)
        rv = np.zeros(self.card, dtype=np.float32)
        delta = np.inf
        while delta > epsilon:
            delta = 0
            for x in range(self.card[0]):
                for y in range(self.card[1]):
                    state = np.array([x, y])
                    if self._is_wall(state):
                        continue
                    old_value = rv[x, y]
                    new_value = 0
                    for action in range(self.num_actions):
                        next_state = self.get_next_state(action, state)
                        reward = self.get_reward(state, next_state)
                        next_state_value = rv[next_state[0], next_state[1]]
                        new_value += policy(action, state) * (reward + self.gamma * next_state_value)
                    rv[x, y] = new_value
                    delta = max(delta, abs(old_value - new_value))
        return rv

    def get_optimal_values(self):

        def get_neighbours(state):
            rv = list()
            if not self._is_wall((state[0], state[1] - 1)):
                rv.append((state[0], state[1] - 1))
            if not self._is_wall((state[0] + 1, state[1])):
                rv.append((state[0] + 1, state[1]))
            if not self._is_wall((state[0], state[1] + 1)):
                rv.append((state[0], state[1] + 1))
            if not self._is_wall((state[0] - 1, state[1])):
                rv.append((state[0] - 1, state[1]))
            return rv

        rv = np.zeros(self.card, dtype=np.float32)
        rv.fill(- np.inf)
        queue = list()
        for x in range(self.card[0]):
            for y in range(self.card[1]):
                if self._is_wall((x, y)):
                    continue
                queue.append((x, y))
        rv[self.terminal_state[0], self.terminal_state[1]] = 0

        while queue:
            queue.sort(key=lambda v: rv[v[0], v[1]])
            x1, y1 = queue.pop()
            reward = 1 if x1 == self.terminal_state[0] and y1 == self.terminal_state[1] else 0
            neighbours = get_neighbours((x1, y1))
            for x2, y2 in neighbours:
                if rv[x2, y2] < reward + self.gamma * rv[x1, y1]:
                    rv[x2, y2] = reward + self.gamma * rv[x1, y1]

        # clean up
        rv[self.terminal_state[0], self.terminal_state[1]] = 0
        for x in range(self.card[0]):
            for y in range(self.card[1]):
                state = np.array([x, y])
                if self._is_wall(state):
                    rv[x, y] = 0

        return rv

    def get_action_values(self, policy, epsilon=1e-6):
        state_values = self.get_state_values(policy, epsilon)
        rv = np.zeros(state_values.shape + (self.num_actions,), dtype=np.float32)
        for x in range(self.card[0]):
            for y in range(self.card[1]):
                state = np.array([x, y])
                if self._is_wall(state):
                    continue
                for action in range(self.num_actions):
                    next_state = self.get_next_state(action, state)
                    reward = self.get_reward(state, next_state)
                    rv[state[0], state[1], action] = reward + self.gamma * \
                        state_values[next_state[0], next_state[1]]
        return rv

    def pretty_print(self, grid):
        rv = ''
        if isinstance(grid[0, 0], np.uint8):
            for row in np.transpose(grid):
                rv += ' '.join([str(v) for v in row]) + '\n'
        elif isinstance(grid[0, 0], np.float32):
            for row in np.transpose(grid):
                rv += ' '.join(['{0:.2f}'.format(v) for v in row]) + '\n'
        else:
            raise TypeError('unsupported type {}'.format(type(grid[0, 0])))
        return rv
