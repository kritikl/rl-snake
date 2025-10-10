from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from snake_env import SnakeEnv

def main():
    env = DummyVecEnv([lambda: SnakeEnv(grid_size=10)])
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        gamma=0.99,
        tensorboard_log=None,
    )

    model.learn(total_timesteps=200_000)
    model.save("snake_ppo_model")

if __name__ == "__main__":
    main()
