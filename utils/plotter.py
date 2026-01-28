import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

TRAIN_DB = "logs/training.db"
EVAL_DB = "logs/evaluation.db"

MODEL_NAMES = ["snake_ppo_seed42", "snake_dqn_seed42"]

#training data
def load_training_data(model_name):
    conn = sqlite3.connect(TRAIN_DB)
    query = """
        SELECT total_steps, reward, score, loops_detected, missed_foods, length
        FROM training_logs
        WHERE model_name = ?
        ORDER BY total_steps
    """
    df = pd.read_sql(query, conn, params=(model_name,))
    conn.close()
    return df


metrics = ["reward", "score", "loops_detected", "missed_foods", "length"]

#training curves
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)

for model in MODEL_NAMES:
    df = load_training_data(model)
    for i, metric in enumerate(metrics):
        # smooth reward and score for readability
        if metric in ["reward", "score", "length"]:
            df[f"smoothed_{metric}"] = df[metric].rolling(window=50, min_periods=1).mean()
            axes[i].plot(df["total_steps"], df[f"smoothed_{metric}"], label=model)
        else:
            axes[i].plot(df["total_steps"], df[metric], label=model)

        axes[i].set_ylabel(metric)

axes[-1].set_xlabel("Total Timesteps")
axes[0].set_title("Training Metrics Over Time")
axes[0].legend()
plt.tight_layout()
plt.show()


#evaluation
def load_eval_data():
    conn = sqlite3.connect(EVAL_DB)
    df = pd.read_sql("SELECT model_name, mean_reward, std_reward FROM evaluation_logs", conn)
    conn.close()
    return df


eval_df = load_eval_data()
eval_df = eval_df[eval_df["model_name"].isin(MODEL_NAMES)]

plt.figure(figsize=(8, 5))
plt.bar(
    eval_df["model_name"],
    eval_df["mean_reward"],
    yerr=eval_df["std_reward"],
    capsize=5
)
plt.ylabel("Mean Evaluation Reward")
plt.title("Evaluation Comparison")
plt.tight_layout()
plt.show()
