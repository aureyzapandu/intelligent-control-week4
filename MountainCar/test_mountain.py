import numpy as np

# Patch untuk NumPy >= 2.0
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model


# Load environment
env = gym.make('MountainCar-v0', render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Load model hasil training
model = load_model(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 4\intelligent-control-week4\dqn_mountain.keras")

scores = []

episodes = 10
for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(200):
        action = np.argmax(model.predict(state, verbose=0)[0])  # greedy
        next_state, reward, terminated, truncated, _ = env.step(action)

        # reward shaping
        reward = reward + (next_state[0] + 0.5) * 10
        if terminated and next_state[0] >= 0.5:
            reward += 100

        done = terminated or truncated
        total_reward += reward
        state = np.reshape(next_state, [1, state_size])

        if done:
            print(f"Test Episode: {e+1}, Score: {total_reward:.0f}")
            break

    scores.append(total_reward)

env.close()

# Plot grafik
plt.figure(figsize=(8,5))
plt.plot(range(1, episodes+1), scores, marker='o')
plt.title("Total Reward per Episode (Testing MountainCar)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
