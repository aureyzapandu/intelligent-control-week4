import numpy as np

# Patch untuk NumPy >= 2.0 agar gym tidak error dengan np.bool8
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import gym
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99          # lebih tinggi, MountainCar butuh long-term reward
        self.epsilon = 1.0         # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()          # online network
        self.target_model = self._build_model()   # target network
        self.update_target_model()                # sync awal

    def _build_model(self):
        model = Sequential([
            Dense(32, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            loss = self.model.train_on_batch(state, target_f)
            total_loss += loss

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return total_loss / batch_size

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]  # = 2 (posisi, kecepatan)
    action_size = env.action_space.n             # = 3 (push left, no push, push right)
    agent = DQNAgent(state_size, action_size)

    episodes = 1000   
    batch_size = 16
    target_update_freq = 10

for e in range(episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # reward shaping
        reward += (next_state[0] + 0.5) * 50
        if done and next_state[0] >= 0.5:
            reward += 100

        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {total_reward:.0f}, Epsilon: {agent.epsilon:.2f}")
            break

    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # update target tiap N episode
    if e % target_update_freq == 0:
        agent.update_target_model()


    # simpan model
    agent.model.save(r"G:\[Data Aureyza\Kuliah Au\SMT 7\8. Prak. Kontrol Cerdas\Week 4\intelligent-control-week4\dqn_mountaincar_target.keras")
