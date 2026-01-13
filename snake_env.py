import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, grid_size=10, render_mode=None):
        super().__init__()
        self.grid_size = grid_size
        self.render_mode = render_mode

        #actions: up, down, left, right
        self.action_space = spaces.Discrete(4)

        #observation: head_x, head_y, food_x, food_y
        self.observation_space = spaces.Box(
            low=0,
            high=grid_size - 1,
            shape=(4,),
            dtype=np.int32
        )

        self.reset()

    def _place_food(self):
        while True:
            fx, fy = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                break

    def _distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self._place_food()
        self.done = False
        self.score = 0
        self.prev_distance = self._distance(self.snake[0], self.food)

        #tracking
        self.visited_states = set()
        self.loops_detected = 0
        self.missed_foods = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        return np.array([head_x, head_y, food_x, food_y], dtype=np.int32)

    def step(self, action):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        move = directions[action]
        head_x, head_y = self.snake[0]
        new_head = (head_x + move[0], head_y + move[1])

        reward = -0.05
        self.done = False

        #loop detection
        state = (new_head, tuple(self.snake), self.food)
        if state in self.visited_states:
            self.loops_detected += 1
            reward -= 1  # penalize loops
        self.visited_states.add(state)

        #missed food detection
        if self._distance(self.snake[0], self.food) == 1:
            if self._distance(new_head, self.food) > 1:  # moved away instead of eating
                self.missed_foods += 1
                reward -= 0.5

        #wall collision
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = -10
            self.done = True
        #self collision
        elif new_head in self.snake:
            reward = -10
            self.done = True
        else:
            self.snake.insert(0, new_head)
            distance = self._distance(new_head, self.food)

            if new_head == self.food:
                reward = 10
                self.score += 1
                self._place_food()
            else:
                if distance < self.prev_distance:
                    reward += 0.5
                else:
                    reward -= 0.3
                self.snake.pop()

            self.prev_distance = distance

        obs = self._get_obs()
        info = {
            "score": self.score,
            "loops_detected": self.loops_detected,
            "missed_foods": self.missed_foods,
        }

        return obs, reward, self.done, False, info

    def render(self):
        if self.render_mode == "human":
            grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            for x, y in self.snake:
                grid[y][x] = "S"
            fx, fy = self.food
            grid[fy][fx] = "F"
            print("\033c", end="")
            for row in grid:
                print(" ".join(row))
            print(f"Score: {self.score}")
