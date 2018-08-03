"""
Deep Q network,

Using:
Tensorflow: 1.0
gym: 0.7.3
"""


import gym
from RL_brain_CNNLSTM import DeepQNetwork
import cv2
import numpy as np
import time
from receiveThread import myThread

env = gym.make('SpaceInvaders-v0')
env = env.unwrapped

print(env.action_space)
# print(env.observation_space)
print(env.observation_space.shape)
print(env.observation_space.high)
print(env.observation_space.low)
print(env.reward_range)

inputImageSize = (100, 80, 1)
# inputImageSize[2] = 1

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  observation_shape=inputImageSize,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,
                  output_graph=True)

total_steps = 0

thread1 = myThread(1, "Thread-1", 1)
thread1.start()

for i_episode in range(100):

    observation = env.reset()
    # 使用opencv做灰度化处理
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    observation = cv2.resize(observation, (inputImageSize[1], inputImageSize[0]))
    total_reward = 0
    while True:
        env.render()
        # observation_, reward, done, info = env.step(env.action_space.sample())
        # print(env.action_space.sample())
        # # observation_, reward, done, info = env.step(4)  # 4是发送子弹 2、3分别是左右
        # if reward > 0:
        #     print(reward)
        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)
        # 使用opencv做灰度化处理
        observation_ = cv2.cvtColor(observation_, cv2.COLOR_BGR2GRAY)
        observation_ = cv2.resize(observation_, (inputImageSize[1], inputImageSize[0]))
        cv2.imshow('obe', observation_)

        RL.store_transition(observation, action, reward, observation_)

        total_reward += reward
        if total_steps > 1000 and total_steps % 2 == 0 and thread1.learn_flag == 1:
            t0 = time.time()
            RL.learn()
            t1 = time.time()
            print("学习一次时间：", t1 - t0)

        if done:
            print('episode: ', i_episode,
                  'total_reward: ', round(total_reward, 2),
                  ' epsilon: ', round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
