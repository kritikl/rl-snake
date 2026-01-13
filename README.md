# Snake-RL-Agent

This project uses **Reinforcement Learning (RL)** to train an agent to play the classic Snake game autonomously.  
The agent learns to survive longer, collect food efficiently, and avoid collisions 


##  Environment and Model


###  Environment

A custom OpenAI Gym environment designed for reinforcement learning on a grid-based Snake game.

- **Observation Space:** Snake head, body, and food coordinates (normalized).  
- **Action Space:** 4 discrete moves — *up, down, left, right*.  
- **Step Function:** Handles movement, growth, collisions, and reward calculation.  
- **Training Environment:** Headless (for speed).  
- **Testing Environment:** Rendered using `turtle`.


### Environment Parameters

The Snake environment is set on a 20×20 grid where the agent controls the snake to collect food and avoid collisions. 
Each step incurs a small living penalty of -0.01 to encourage efficient play. 
Eating food gives a positive reward of +10, while collisions with the wall or the snake's own body incur a -10 penalty. 

These parameters balance exploration and risk, helping the agent learn both survival and effective food collection strategies.



### Action Space

The agent can choose from four discrete actions corresponding to the snake's movement directions: up, down, left, and right. 
Each action moves the snake's head by one grid cell in the chosen direction. 
The action space is designed to be simple yet sufficient for learning efficient navigation and food collection on the grid.


### Observation Space

The observation vector consists of:
1. Snake head position  
2. Food position  
3. Relative distance (optionally wall/self proximity)

Example (minimal 4-value observation):
[snake_x, snake_y, food_x, food_y]

### Model

The model uses Proximal Policy Optimization (PPO) from Stable Baselines3, with a multi-layer perceptron (MLP) policy.

### Reward Function

The agent receives a positive reward of +10 when it eats the food. 
Each time step incurs a small negative reward of -0.05 to encourage shorter paths.  
The agent is rewarded with +0.5 for moving closer to the food and penalized with -0.3 if it moves further away. 
Colliding with the walls or itself results in a large negative reward of -10 and ends the episode. 

### Training Details

The snake agent is trained using the Proximal Policy Optimization (PPO) algorithm from Stable Baselines3 with an MLP policy. 
The environment is vectorized across multiple processes for faster training, and a total of 2,000,000 timesteps are used to optimize the policy. 


