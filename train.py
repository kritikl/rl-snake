import os
import numpy as np

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from snake_env import SnakeEnv
from logs.logger import TrainingLogger, EvaluationLogger
from utils.seeding import seeding


class TrainingLoggingCallback(BaseCallback):
    def __init__(self, model_name, train_logger, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.train_logger = train_logger

        self.episode = 0
        self.total_steps = 0
        self.ep_reward = 0.0
        self.ep_length = 0

    def _on_step(self) -> bool:
        self.total_steps += 1
        self.ep_reward += float(self.locals["rewards"][0])
        self.ep_length += 1

        if self.locals["dones"][0]:
            info = self.locals["infos"][0]

            self.train_logger.log(
                self.model_name,
                self.episode,
                self.ep_reward,
                info.get("score", 0),
                info.get("loops_detected", 0),
                info.get("missed_foods", 0),
                self.ep_length,
                self.total_steps,
            )

            self.ep_reward = 0.0
            self.ep_length = 0
            self.episode += 1

        return True


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    SEED = 42
    ALGO = "ppo"
    TOTAL_TIMESTEPS = 2_000_000

    seeding(SEED)

    model_name = f"snake_{ALGO}_seed{SEED}"

    train_env = SnakeEnv(seed=SEED)
    eval_env = SnakeEnv(seed=SEED)

    if ALGO == "ppo":
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            learning_rate=3e-4,
            seed=SEED,
            verbose=1,
        )

    elif ALGO == "dqn":
        model = DQN(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=10_000,
            seed=SEED,
            verbose=1,
        )
        assert train_env.action_space.n == 4

    else:
        raise ValueError("Unsupported algorithm")

    train_logger = TrainingLogger()
    eval_logger = EvaluationLogger()

    callback = TrainingLoggingCallback(
        model_name=model_name,
        train_logger=train_logger,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
    )

    model.save(f"models/{model_name}")
    train_logger.close()

    episodes = 20
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = eval_env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

    mean_reward = float(np.mean(episode_rewards))
    std_reward = float(np.std(episode_rewards))

    eval_logger.log(
        model_name=model_name,
        mean_reward=mean_reward,
        std_reward=std_reward,
        episodes_evaluated=episodes,
    )

    eval_env.close()
    eval_logger.close()
