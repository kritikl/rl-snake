import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnv
from logs.logger import TrainingLogger, EvaluationLogger
import numpy as np


from stable_baselines3.common.callbacks import BaseCallback

class TrainingLoggingCallback(BaseCallback):
    def __init__(self, model_name, train_logger, verbose=0):
        super().__init__(verbose)
        self.model_name = model_name
        self.train_logger = train_logger

        # episode tracking
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
                self.total_steps
            )

            #reset trackers
            self.ep_reward = 0.0
            self.ep_length = 0
            self.episode += 1

        return True



if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    model_name = "snake_ppo_v2"

    train_env = SnakeEnv()
    eval_env = SnakeEnv()

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        verbose=1
    )

    train_logger = TrainingLogger()
    eval_logger = EvaluationLogger()

    callback = TrainingLoggingCallback(
        model_name=model_name,
        train_logger=train_logger 
    )

    #training
    model.learn(
        total_timesteps=2_000_000,
        callback=callback
    )

    model.save(f"models/{model_name}")
    train_logger.close()

    # evaluation
    episodes = 20
    episode_rewards = []

    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        total_reward = 0

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
        episodes_evaluated=episodes
    )

    eval_env.close()
    eval_logger.close()
