__author__ = 'think see'
__date__ = '2019/5/30'
# These codes reference the http://lxiao.xmu.edu.cn/
from mobile_crowdsensing_games_for_individual_privacy.dqn.cnn.cnn import DQN
from mobile_crowdsensing_games_for_individual_privacy.agent import MCSAgent, UserAgent
from mobile_crowdsensing_games_for_individual_privacy.q_learning.qlearning import QLearningMultiUser
from mobile_crowdsensing_games_for_individual_privacy.utils import get_saved_matrix_mulit, get_saved_matrix_single
from mobile_crowdsensing_games_for_individual_privacy.utils import save_to_txt_multi, save_to_txt_single
from mobile_crowdsensing_games_for_individual_privacy.utils import plot_result_multi, plot_result_single
from mobile_crowdsensing_games_for_individual_privacy.param import *
from tqdm import tqdm
import numpy as np
import os
# create the document which can different the img and data.
if not os.path.exists('img'):
    os.makedirs('img')

if not os.path.exists('data'):
    os.makedirs('data')


def game_2user(n_user=2, func=1):
    agent_mcs = MCSAgent(np.arange(1, 5, 0.5))
    agent_user1 = UserAgent(np.arange(1, 21, 2), np.arange(0.1, 1.1, 0.1))
    agent_user2 = UserAgent(np.arange(1, 21, 2), np.arange(0.1, 1.1, 0.1))
    # input_length : [user1_state, user2_state]
    model_mcs = DQN(n_user, agent_mcs.n_actions,
                    memory_capacity=MEMORY_CAPACITY, window=WINDOW,
                    gamma=GAMMA, eps_start=EPS_START,
                    eps_end=EPS_END, anneal_step=ANNEAL_STEP,
                    learning_begin=LEARNING_BEGIN
                    )
    # input_length ; [user1_state, payment]
    model_user1 = DQN(n_user, agent_user1.get_actions_len(),
                      memory_capacity=MEMORY_CAPACITY, window=WINDOW,
                      gamma=GAMMA, eps_start=EPS_START,
                      eps_end=EPS_END, anneal_step=ANNEAL_STEP,
                      learning_begin=LEARNING_BEGIN
                      )
    model_user2 = DQN(n_user, agent_user2.get_actions_len(),
                      memory_capacity=MEMORY_CAPACITY, window=WINDOW,
                      gamma=GAMMA, eps_start=EPS_START,
                      eps_end=EPS_END, anneal_step=ANNEAL_STEP,
                      learning_begin=LEARNING_BEGIN
                      )

    matrix_utility_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user1 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user2 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_action_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user1 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user2 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_action_mcs_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user1_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user2_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_aggregate_error = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    for episode in tqdm(range(MAX_EPISODE)):
        model_mcs.reset()
        model_user1.reset()
        model_user2.reset()
        cur_mcs_action_index, cur_mcs_action = agent_mcs.init_action_and_index_by_random()
        cur_user1_action_index, cur_user1_action = agent_user1.single_init_action_and_index_by_random()
        cur_user2_action_index, cur_user2_action = agent_user2.single_init_action_and_index_by_random()
        for step in range(MAX_STEP):
            # 1. composite state
            cur_mcs_state = np.array([cur_user1_action_index, cur_user2_action])
            cur_user1_state = np.array([cur_user1_action, cur_mcs_action])
            cur_user2_state = np.array([cur_user2_action, cur_mcs_action])
            # 2. select action with dqn
            next_mcs_action_index = model_mcs.select_action(cur_mcs_state)
            next_mcs_action = agent_mcs.get_action_value(next_mcs_action_index)

            next_user1_action_index = model_user1.select_action(cur_user1_state)
            next_user1_action = agent_user1.get_action_value(next_user1_action_index)

            next_user2_action_index = model_user2.select_action(cur_user2_state)
            next_user2_action = agent_user2.get_action_value(next_user2_action_index)
            # 3. game
            # 3.1 leader multi-cast the payment list
            payment = agent_mcs.get_payments(cur_mcs_action, agent_user1.get_actions_len(), PAYMENT_ACC)

            if func == 1:
                aggregate_error, mcs_utility_all = agent_mcs.get_mcs_utility_reciprocal(
                    user_action_list=[cur_user1_action, cur_user2_action],
                    data_range=DATA_RANGE,
                    confidence_level=CONFIDENCE_LEVEL,
                    n_user=n_user)
            elif func == 2:
                aggregate_error, mcs_utility_all = agent_mcs.get_mcs_utility_percentage(
                    user_action_list=[cur_user1_action, cur_user2_action],
                    data_range=DATA_RANGE,
                    confidence_level=CONFIDENCE_LEVEL,
                    n_user=n_user)
            else:
                raise NameError('function is\'t exist.')

            r_mcs = mcs_utility_all - payment[cur_user1_action_index] - payment[cur_user2_action_index]

            # 3.2 followers get utility
            r_user1 = payment[cur_user1_action_index] - agent_user1.get_cost_value(cur_user1_action_index) * cur_user1_action
            r_user2 = payment[cur_user2_action_index] - agent_user2.get_cost_value(cur_user2_action_index) * cur_user2_action
            # 4. get next state
            next_mcs_state = np.array([next_user1_action_index, next_user2_action_index])
            next_user1_state = np.array([next_mcs_action_index, next_user1_action_index])
            next_user2_state = np.array([next_mcs_action_index, next_user2_action_index])
            # 5. optimize model
            model_mcs.optimize_model(cur_mcs_state, next_mcs_state, cur_mcs_action_index, r_mcs, batch_size=BATCH_SIZE)
            model_user1.optimize_model(cur_user1_state, next_user1_state, cur_user1_action_index, r_user1, batch_size=BATCH_SIZE)
            model_user2.optimize_model(cur_user2_state, next_user2_state, cur_user2_action_index, r_user2, batch_size=BATCH_SIZE)

            # 6. record parameter
            # utility
            matrix_utility_mcs[episode][step] = r_mcs
            matrix_utility_user1[episode][step] = r_user1
            matrix_utility_user2[episode][step] = r_user2
            # action
            matrix_action_mcs[episode][step] = cur_mcs_action
            matrix_action_user1[episode][step] = cur_user1_action
            matrix_action_user2[episode][step] = cur_user2_action
            # action index
            matrix_action_mcs_index[episode][step] = cur_mcs_action_index
            matrix_action_user1_index[episode][step] = cur_user1_action_index
            matrix_action_user2_index[episode][step] = cur_user2_action_index
            # aggregate error
            matrix_aggregate_error[episode][step] = aggregate_error

            # 7. update state
            cur_mcs_action, cur_mcs_action_index, cur_mcs_state = next_mcs_action, next_mcs_action_index, next_mcs_state
            cur_user1_action, cur_user1_action_index, cur_user1_state = next_user1_action, next_user1_action_index, next_user1_state
            cur_user2_action, cur_user2_action_index, cur_user2_state = next_user2_action, next_user2_action_index, next_user1_state

    # 8. persistence parameter
    # data, utility, greedy, mcs, max_step, n_user=120, function='reciprocal'
    if func == 1:
        f = 'reciprocal'
    elif func == 2:
        f = 'percentage'
    else:
        raise NameError('function is\'t exist.')

    save_to_txt_single(matrix_utility_mcs, 'utility', 'egreedy', 'mcs', MAX_STEP, n_user, f)
    save_to_txt_single(matrix_utility_user1, 'utility', 'egreedy', 'user1', MAX_STEP, n_user, f)
    save_to_txt_single(matrix_utility_user2, 'utility', 'egreedy', 'user2', MAX_STEP, n_user, f)

    save_to_txt_single(matrix_action_mcs_index, 'action', 'egreedy', 'mcs', MAX_STEP, n_user, f)
    save_to_txt_single(matrix_action_user1_index, 'action', 'egreedy', 'user1', MAX_STEP, n_user, f)
    save_to_txt_single(matrix_action_user2_index, 'action', 'egreedy', 'user2', MAX_STEP, n_user, f)

    save_to_txt_single(matrix_aggregate_error, 'aggregate-error', 'egreedy', 'mcs', MAX_STEP, n_user, f)

    plot_result_single(matrix_utility_mcs,
                       matrix_utility_user1,
                       matrix_utility_user2,
                       matrix_aggregate_error,
                       matrix_action_mcs_index,
                       matrix_action_user1_index,
                       matrix_action_user2_index,
                       MAX_EPISODE,
                       MAX_STEP,
                       'dqn\\cnn',
                       'single',
                       f,
                       n_user)


def game_n_user(n_user=60, func=1):
    agent_mcs = MCSAgent(np.arange(1, 5, 0.5))
    agent_users = UserAgent(np.arange(1, 21, 2), np.arange(0.1, 1.1, 0.1), n_user)
    # the number of the privacy selected by users,
    # so that the length is the length of user's action.
    dqn_mcs = DQN(agent_users.get_actions_len(),
                  agent_mcs.n_actions,
                  memory_capacity=MEMORY_CAPACITY,
                  window=WINDOW,
                  gamma=GAMMA,
                  eps_start=EPS_START,
                  eps_end=EPS_END,
                  anneal_step=ANNEAL_STEP,
                  learning_begin=LEARNING_BEGIN)

    qtable_users = QLearningMultiUser(n_user, actions=agent_users.get_actions_index())
    matrix_utility_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_action_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_action_mcs_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user_index = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_aggregate_error = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    for episode in tqdm(range(MAX_EPISODE)):
        # clear mcs and user
        dqn_mcs.reset()
        qtable_users.clear_all()
        # init action
        cur_mcs_action_index, cur_mcs_action = agent_mcs.init_action_and_index_by_random()
        cur_user_action_index, cur_user_action = agent_users.multi_init_action_and_index_by_random()
        for step in range(MAX_STEP):
            # 1. composite state
            # 1.1 server state. number of actions selected by all clients for each action
            cur_user_action_num = agent_users.zero_actions_len()
            for idx in range(n_user):
                cur_user_action_num[int(cur_user_action_index[idx])] += 1
            cur_mcs_state = np.array(cur_user_action_num)
            # 1.2 client state. action of each client and server
            cur_user_state = agent_users.zero_user_state_len()
            for idx in range(n_user):
                cur_user_state[idx] = str([cur_mcs_action_index, cur_user_action_index[idx]])
                # print(cur_user_state[idx])

            # 2. select action with q table
            # 2.1 server
            next_mcs_action_index = dqn_mcs.select_action(cur_mcs_state)
            next_mcs_action = agent_mcs.get_action_value(cur_mcs_action_index)
            # 2.2 client
            next_user_action_index = agent_users.zero_user_len(tt=np.int32)
            next_user_action = agent_users.zero_user_len(tt=np.float32)
            for idx in range(n_user):
                next_user_action_index[idx] = qtable_users.select_action(int(idx), cur_user_state[idx], 'e-greedy')
                next_user_action[idx] = agent_users.get_action_value(next_user_action_index[idx])

            # 3. game
            # 3.1 leader multicast the payment list
            payment = agent_mcs.get_payments(next_mcs_action, agent_users.get_actions_len(), 0.5)

            if func == 1:
                aggregate_error, mcs_utility_all = agent_mcs.get_mcs_utility_reciprocal(
                    user_action_list=cur_user_action,
                    data_range=DATA_RANGE,
                    confidence_level=CONFIDENCE_LEVEL,
                    n_user=n_user)
            elif func == 2:
                aggregate_error, mcs_utility_all = agent_mcs.get_mcs_utility_percentage(
                    user_action_list=cur_user_action,
                    data_range=DATA_RANGE,
                    confidence_level=CONFIDENCE_LEVEL,
                    n_user=n_user)
            else:
                raise NameError('function is\'t exist.')

            r_mcs = agent_mcs.get_mcs_reward(mcs_utility_all, next_user_action_index, payment)

            # 3.2 followers get utility
            r_user = agent_users.zero_user_len(tt=np.float32)
            for idx in range(n_user):
                r_user[idx] = payment[next_user_action_index[idx]] - agent_users.get_cost_value(
                    next_user_action_index[idx]) * next_user_action[idx]

            # 4. get next state
            next_user_action_num = agent_users.zero_user_len(tt=np.int32)
            for idx in range(n_user):
                next_user_action_num[next_user_action_index[idx]] += 1

            # 4.1  server state. number of actions selected by all clients for each action
            next_mcs_state = np.array(next_user_action_num)
            # 4.2 client state. action of each client and server
            next_user_state = agent_users.zero_user_state_len()
            for idx in range(n_user):
                next_user_state[idx] = str([next_mcs_action_index, next_user_action_index[idx]])

            # 5. learning
            # 5.1 learning through q learning
            for idx in range(n_user):
                qtable_users.learn(idx, cur_user_state[idx], cur_user_action_index[idx], r_user[idx],
                                   next_user_state[idx])  # s, a, r, s_
            # 5.2 learning through cnn
            dqn_mcs.optimize_model(cur_mcs_state, next_mcs_state, cur_mcs_action_index, r_mcs, batch_size=BATCH_SIZE)

            # 6. record parameter
            # 6.1 client
            for idx in range(n_user):
                # utility
                matrix_utility_user[episode][step][idx] = r_user[idx]
                # action
                matrix_action_user[episode][step][idx] = cur_user_action[idx]
                # action index
                matrix_action_user_index[episode][step][idx] = cur_user_action_index[idx]
            # 6.2 server
            matrix_action_mcs_index[episode][step] = cur_mcs_action_index
            matrix_action_mcs[episode][step] = cur_mcs_action
            matrix_utility_mcs[episode][step] = r_mcs
            # aggregate error
            matrix_aggregate_error[episode][step] = aggregate_error

            # 7. update state
            cur_mcs_action, cur_mcs_action_index, cur_mcs_state = next_mcs_action, next_mcs_action_index, next_mcs_state
            cur_user_action, cur_user_action_index, cur_user_state = next_user_action, next_user_action_index, next_user_state

    # 8. persistence parameter
    # data, utility, greedy, mcs, max_step, n_user=120, function='reciprocal'
    if func == 1:
        f = 'reciprocal'
    elif func == 2:
        f = 'percentage'
    else:
        raise NameError('function is\'t exist.')
    save_to_txt_multi(matrix_utility_mcs, 'utility', 'egreedy', 'mcs', MAX_STEP, n_user, f)

    matrix_utility_user = np.sum(matrix_utility_user, axis=2) / n_user
    save_to_txt_multi(matrix_utility_user, 'utility', 'egreedy', 'user', MAX_STEP, n_user, f)

    save_to_txt_multi(matrix_action_mcs_index, 'action', 'egreedy', 'mcs', MAX_STEP, n_user, f)

    matrix_action_user_index = np.max(matrix_action_user_index, axis=2)
    save_to_txt_multi(matrix_action_user_index, 'action', 'egreedy', 'user', MAX_STEP, n_user, f)

    save_to_txt_multi(matrix_aggregate_error, 'aggregate-error', 'egreedy', 'mcs', MAX_STEP, n_user, f)

    plot_result_multi(matrix_utility_mcs,
                      matrix_utility_user,
                      matrix_aggregate_error,
                      matrix_action_mcs_index,
                      matrix_action_user_index,
                      MAX_EPISODE,
                      MAX_STEP,
                      'dqn\\cnn',
                      'multi',
                      f,
                      n_user)


if __name__ == '__main__':
    game_n_user(n_user=60, func=1)
    game_2user(n_user=2, func=1)
