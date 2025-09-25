import gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent

env = gym.make('CartPole-v1', render_mode="human")  # render di window
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Inisialisasi agen (pakai model terlatih)
agent = DQNAgent(state_size, action_size)
agent.model.load_weights(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 4\intelligent-control-week4\dqn_cartpole_model_100.h5")
agent.epsilon = 0.01  # Minimalkan eksplorasi saat testing

scores = []  # untuk menyimpan total reward tiap episode

episodes = 20
for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        action = np.argmax(agent.model.predict(state, verbose=0)[0])  # greedy
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = np.reshape(next_state, [1, state_size])

        if done:
            print(f"Test Episode: {e+1}, Score: {total_reward}")
            break

    scores.append(total_reward)

env.close()

# --- Plot grafik ---
plt.figure(figsize=(8,5))
plt.plot(range(1, episodes+1), scores, marker='o')
plt.title("Total Reward per Episode (Testing)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.xticks(range(1, episodes+1))
plt.grid(True)
plt.show()
