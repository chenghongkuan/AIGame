# -*- coding: UTF-8 -*-

"""
这个是DQN大脑部分
使用Keras建造神经网络
可以看我的CSDN：https://blog.csdn.net/u012465304
"""
# 如果要用tensorflow就把下面这个注释给删掉，接下来我们先用tensorflow
# import os
# os.environ['KERAS_BACKEND']='theano'
import numpy as np
# 按顺序建立的神经网络
from keras.models import Sequential
# dense是全连接层，这里选择你要用的神经网络层参数
from keras.layers import LSTM, TimeDistributed, Dense, Activation
# 选择优化器
from keras.optimizers import Adam, RMSprop
# 画图
from keras.utils import plot_model

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
            first_layer_neurno=4,
            second_layer_neurno=1
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.first_layer_neurno = first_layer_neurno
        self.second_layer_neurno = second_layer_neurno

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # print("1")
        if output_graph:
            print("输出图像")
            plot_model(self.model_eval, to_file='model1.png')
            plot_model(self.model_target, to_file='model2.png')

        # 记录cost然后画出来
        self.cost_his = []

    def _build_net(self):
        # ------------------ 建造估计层 ------------------
        # 因为神经网络在这个地方只是用来输出不同动作对应的Q值，最后的决策是用Q表的选择来做的
        # 所以其实这里的神经网络可以看做是一个线性的，也就是通过不同的输入有不同的输出，而不是确定类别的几个输出
        # 这里我们先按照上一个例子造一个两层每层单个神经元的神经网络
        self.model_eval = Sequential([
            # 输入 并且给每一个神经元配一个激活函数
            Dense(self.first_layer_neurno, input_dim=self.n_features, activation='relu'),
            # Activation('relu'),
            # Dense(1, activation='tanh'),
            # 输出
            Dense(self.n_actions),
        ])
        # 选择rms优化器，输入学习率参数
        rmsprop = RMSprop(lr=self.lr, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model_eval.compile(loss='mse',
                            optimizer=rmsprop,
                            metrics=['accuracy'])

        # ------------------ 构建目标神经网络 ------------------
        # 目标神经网络的架构必须和估计神经网络一样，但是不需要计算损失函数
        self.model_target = Sequential([
            # 输入 并且给每一个神经元配一个激活函数
            Dense(self.first_layer_neurno, input_dim=self.n_features, activation='relu'),
            # Activation('relu'),
            # Dense(1, activation='tanh'),
            # 输出
            Dense(self.n_actions),
        ])

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        print("T")
        print(transition)
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # 插入一个新的维度 矩阵运算时需要新的维度来运算
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # 向前反馈，得到每一个当前状态每一个action的Q值
            # 这里使用估计网络，也就是要更新参数的网络
            # 然后选择最大值,这里的action是需要执行的action
            # print(observation)
            actions_value = self.model_eval.predict(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            # print(action)
        return action

    def learn(self):
        # 经过一定的步数来做参数替换
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.model_target.set_weights(self.model_eval.get_weights())
            print('\ntarget_params_replaced\n')

        # 随机取出记忆
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 这里需要得到估计值加上奖励 成为训练中损失函数的期望值
        # q_next是目标神经网络的q值，q_eval是估计神经网络的q值
        # q_next是用现在状态得到的q值 q_eval是用这一步之前状态得到的q值
        # print(batch_memory[:, -self.n_features:])
        q_next = self.model_target.predict(batch_memory[:, -self.n_features:], batch_size=self.batch_size)
        q_eval = self.model_eval.predict(batch_memory[:, :self.n_features], batch_size=self.batch_size)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
          假如在这个 batch 中, 我们有2个提取的记忆, 根据每个记忆可以生产3个 action 的值:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        然后根据 memory 当中的具体 action 位置来修改 q_target 对应 action 上的值:
        比如在:
            记忆 0 的 q_target 计算值是 -1, 而且我用了 action 0;
            记忆 1 的 q_target 计算值是 -2, 而且我用了 action 2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        所以 (q_target - q_eval) 就变成了:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        最后我们将这个 (q_target - q_eval) 当成误差, 反向传递会神经网络.
        所有为 0 的 action 值是当时没有选择的 action, 之前有选择的 action 才有不为0的值.
        我们只反向传递之前选择的 action 的值,
        """

        # 训练估计网络，用的是当前观察值训练，并且训练选择到的数据加奖励训练而不是没选择的
        self.cost = self.model_eval.train_on_batch(batch_memory[:, :self.n_features], q_target)

        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



