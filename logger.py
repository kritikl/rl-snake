import csv
import os
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir="logs", filename_prefix="training_log"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{filename_prefix}_{timestamp}.csv")

        with open(self.log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Step", "Reward", "Total_Reward", "Done"])

    def log_step(self, episode, step, reward, total_reward, done):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, reward, total_reward, done])

    def log_summary(self, episode, total_reward, steps):
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, steps, "", total_reward, "Summary"])
