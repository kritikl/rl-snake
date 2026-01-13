from turtle import Screen, Turtle
import time
import numpy as np
from snake_env import SnakeEnv
from stable_baselines3 import PPO

#screen
screen = Screen()
screen.setup(width=600, height=600)
screen.bgcolor("black")
screen.title("Snake")
screen.tracer(0)

#env + model
env = SnakeEnv(grid_size=10)
model = PPO.load("rl_snake/models/snake_ppo_v2", env=env)

CELL_SIZE = 40  # 10x10 grid in 600x600
OFFSET = - (env.grid_size * CELL_SIZE) // 2

#turtle
snake_turtles = []
food_turtle = Turtle("circle")
food_turtle.color("red")
food_turtle.penup()

score_display = Turtle()
score_display.hideturtle()
score_display.color("white")
score_display.penup()
score_display.goto(0, 250)

game_over_display = Turtle()
game_over_display.hideturtle()
game_over_display.color("red")
game_over_display.penup()

def grid_to_pixel(x, y):
    """Convert grid coords (0..grid_size-1) → pixel coords."""
    return OFFSET + x * CELL_SIZE, OFFSET + y * CELL_SIZE

def update_snake_visual():
    """Update snake body visuals."""
    global snake_turtles

    while len(snake_turtles) > len(env.snake):
        t = snake_turtles.pop()
        t.hideturtle()

    while len(snake_turtles) < len(env.snake):
        t = Turtle("square")
        t.color("white")
        t.penup()
        snake_turtles.append(t)

    for seg, (x, y) in zip(snake_turtles, env.snake):
        seg.goto(grid_to_pixel(x, y))

def update_food_visual():
    fx, fy = env.food
    food_turtle.goto(grid_to_pixel(fx, fy))

def update_score_display():
    score_display.clear()
    score_display.write(f"Score: {env.score}", align="center", font=("Courier", 24, "bold"))

def show_game_over():
    game_over_display.goto(0, 0)
    game_over_display.write(f"GAME OVER\nFinal Score: {env.score}",
                            align="center", font=("Courier", 28, "bold"))

#initialization
update_snake_visual()
update_food_visual()
update_score_display()
screen.update()

#run
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    update_snake_visual()
    update_food_visual()
    update_score_display()
    screen.update()
    time.sleep(0.15)

#game over
show_game_over()
screen.update()

screen.mainloop()
