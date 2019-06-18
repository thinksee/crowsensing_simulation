import torch
import random
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from collections import namedtuple

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor
Tensor = FloatTensor

random.seed(2000)

# 'state' -- numpy, 'next_state' -- numpy, 'action' -- int, 'reward' -- float
# better than tuple
Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward'))

# 'net_input' -- numpy, 'next_net_input' -- numpy, 'action' -- int, 'reward' -- float
Transition_chain = namedtuple('Transition_chain',
                              ('net_input', 'next_net_input', 'action', 'reward'))


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)


class ReplayMemory(object):
    def __init__(self, capacity, window, input_length):
        self.__window = window
        self.__input_length = input_length
        self.__capacity = capacity
        self.__memory = []
        self.__memory_chain = []

    def __len__(self):
        return len(self.__memory_chain)

    def reset(self):
        self.__memory = []
        self.__memory_chain = []

    def get_net_input(self, state):
        memory_length = len(self.__memory)
        if memory_length <= self.__window:
            return None
        else:
            net_input = []
            # get the tail element of list
            for i in range(memory_length - self.__window, memory_length):
                # +=: it requires that the object on the right be iterable, such as tuples, lists and even dictionaries.
                # append: it append th object on the right as a whole to the end of the list.
                net_input += self.__memory[i].state.tolist()
                net_input.append(self.__memory[i].action)

            net_input += state.tolist()

            net_input = np.array(net_input).reshape(-1)  # 转为1行
            return net_input

    def push(self, state, next_state, action, reward):
        net_input = self.get_net_input(state)  # the size is window size
        self.__memory.append(Transition(state, next_state, action, reward))
        if len(self.__memory) > self.__capacity:
            # default remove the last element, and return the value.
            self.__memory.pop(0)
        next_net_input = self.get_net_input(next_state)

        if net_input is not None and next_net_input is not None:
            self.__memory_chain.append(Transition(net_input, next_net_input, action, reward))
            if len(self.__memory_chain) > self.__capacity:
                self.__memory_chain.pop(0)

        return net_input, next_net_input

    def sample(self, batch_size):
        return random.sample(self.__memory_chain, batch_size)


class DQNModel(nn.Module):

    def __init__(self, input_length, num_action):
        super(DQNModel, self).__init__()

        self.num_action = num_action
        self.cov1 = nn.Conv1d(1, 20, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(20)
        self.cov2 = nn.Conv1d(20, 40, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(40)
        # convolution kernel. first layer is 3x3 and second layer is 2x2
        self.fc1 = nn.Linear(40 * (input_length + 1 - 3 + 1 - 2), 180)
        self.fc2 = nn.Linear(180, self.num_action)

    def forward(self, x):  # implement the forward, the backward auto implement with the autograd mechanism
        x = x.view(x.size(0), -1, x.size(-1))
        x = functional.leaky_relu(self.bn1(self.cov1(x)))
        x = functional.leaky_relu(self.bn2(self.cov2(x)))
        x = x.view(x.size(0), -1)

        x = functional.leaky_relu(self.fc1(x))
        x = functional.leaky_relu(self.fc2(x))

        return x

    def reset(self):
        self.cov1.reset_parameters()
        self.bn1.reset_parameters()
        self.cov2.reset_parameters()
        self.bn2.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2 = nn.Linear(180, self.num_action)
        self.fc2.reset_parameters()


class DQN(object):

    def __init__(self,
                 input_length,
                 num_action,
                 memory_capacity,
                 window,
                 gamma=0.5,
                 eps_start=1.0,
                 eps_end=0.1,
                 anneal_step=100,
                 learning_begin=50):
        self.num_action = num_action
        self.memory = ReplayMemory(memory_capacity, window, input_length)
        self.gamma = gamma
        self.eps_end = eps_end
        self.eps_start = eps_start
        self.anneal_step = anneal_step
        self.learning_begin = learning_begin

        self.model = DQNModel((input_length + 1) * window + input_length, num_action)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-7)
        self.steps_done = 0

    def reset(self):
        self.memory.reset()
        self.steps_done = 0
        self.model.reset()

    def select_action(self, state):

        net_input = self.memory.get_net_input(state)

        if net_input is not None:
            sample = random.random()
            eps_tmp = self.eps_start - \
                      (self.steps_done - self.learning_begin) * (self.eps_start - self.eps_end) / self.anneal_step
            eps_tmp = min(self.eps_start, eps_tmp)
            eps_threshold = max(self.eps_end, eps_tmp)
            self.steps_done += 1

            if sample > eps_threshold:
                # print(Variable(torch.from_numpy(net_input.reshape(1, -1)).float()))
                _, action_ind = self.model(Variable(torch.from_numpy(net_input.reshape(1, -1)).float())).data.max(dim=1)

                return int(action_ind.item())

            else:
                return int(random.randrange(self.num_action))

        else:
            return int(random.randrange(self.num_action))

    def optimize_model(self, state, next_state, action, reward, batch_size=10):

        net_input, next_net_input = self.memory.push(state, next_state, action, reward)

        if len(self.memory) < batch_size:
            return

        experience = self.memory.sample(batch_size)  # sample. four tuples have window size
        batch = Transition_chain(*zip(*experience))  # four elements in an aggregate tuple
        next_states_batch = Variable(torch.cat([FloatTensor(batch.next_net_input)]))  #
        state_batch = Variable(torch.cat([FloatTensor(batch.net_input)]))

        action_batch = Variable(torch.cat([LongTensor(batch.action)]).view(-1, 1))  # put multiple rows together one row
        reward_batch = Variable(torch.cat([FloatTensor(batch.reward)]))
        # extract elements according to row pattern with action_batch as index
        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_action_values = self.model(next_states_batch).max(1)[0]

        # next_state_action_values.volatile = False

        expected_state_action_values = (next_state_action_values * self.gamma) + reward_batch  # calculate the reward

        loss2 = functional.smooth_l1_loss(state_action_values.squeeze(), expected_state_action_values)  # original loss
        self.optimizer.zero_grad()  # set the parameter gradient to 0
        loss2.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



