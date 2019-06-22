__author__ = 'think see'
__date__ = '2019/3/14'

from mobile_crowdsensing_games_for_individual_privacy.q_learning.qlearning import QLearningSingleUser, QLearningMultiUser
from mobile_crowdsensing_games_for_individual_privacy.agent import UserAgent, MCSAgent
from mobile_crowdsensing_games_for_individual_privacy.utils import get_saved_matrix_single, get_saved_matrix_mulit
from mobile_crowdsensing_games_for_individual_privacy.utils import save_to_txt_single, save_to_txt_multi
from mobile_crowdsensing_games_for_individual_privacy.utils import plot_result_single, plot_result_multi
from mobile_crowdsensing_games_for_individual_privacy.param import *
from tqdm import tqdm
import os
import numpy as np


if not os.path.exists('img'):
    os.makedirs('img')

if not os.path.exists('data'):
    os.makedirs('data')


def game_2user(n_user=N_USER_SINGLE, func=FUNC):
    """
    :param n_user:
    :param func:  reciprocal or percentage, where function=1 represent reciprocal and
      function = 2 represent percentage.
    :return:
    """
    agent_mcs = MCSAgent(MCS_ACTION)
    agent_user1 = UserAgent(USER_ACTION, USER_COST)
    agent_user2 = UserAgent(USER_ACTION, USER_COST)
    mcs_qtable = QLearningSingleUser(actions=agent_mcs.get_actions_index())
    user1_qtable = QLearningSingleUser(actions=agent_user1.get_actions_index())
    user2_qtable = QLearningSingleUser(actions=agent_user2.get_actions_index())

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
        # clear q-table
        mcs_qtable.clear()
        user1_qtable.clear()
        user2_qtable.clear()
        # init action by random
        pre_mcs_action_index, pre_mcs_action = agent_mcs.init_action_and_index_by_random()
        pre_user1_action_index, pre_user1_action = agent_user1.single_init_action_and_index_by_random()
        pre_user2_action_index, pre_user2_action = agent_user2.single_init_action_and_index_by_random()
        for step in range(MAX_STEP):
            # 1. composite state
            cur_mcs_state = str([pre_user1_action_index, pre_user2_action_index])
            cur_user1_state = str([pre_mcs_action_index, pre_user1_action_index])
            cur_user2_state = str([pre_mcs_action_index, pre_user2_action_index])

            # 2. select action with q learning
            cur_user1_action_index = user1_qtable.select_action(cur_user1_state, 'e-greedy')
            cur_user1_action = agent_user1.get_action_value(cur_user1_action_index)

            cur_user2_action_index = user2_qtable.select_action(cur_user2_state, 'e-greedy')
            cur_user2_action = agent_user2.get_action_value(cur_user2_action_index)

            cur_mcs_action_index = mcs_qtable.select_action(cur_mcs_state, 'e-greedy')
            cur_mcs_action = agent_mcs.get_action_value(cur_mcs_action_index)
            # 3. game
            # 3.1 leader multicast the payment list
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

            cur_r_mcs = mcs_utility_all - payment[cur_user1_action_index] - payment[cur_user2_action_index]

            # 3.2 followers get utility
            cur_r_user1 = payment[cur_user1_action_index] - agent_user1.get_cost_value(cur_user1_action_index) * cur_user1_action
            cur_r_user2 = payment[cur_user2_action_index] - agent_user2.get_cost_value(cur_user2_action_index) * cur_user2_action

            # 4. get next state
            next_mcs_state = str([cur_user1_action_index, cur_user2_action_index])
            next_user1_state = str([cur_mcs_action_index, cur_user1_action_index])
            next_user2_state = str([cur_mcs_action_index, cur_user2_action_index])

            # 5. learning
            user1_qtable.learn(cur_user1_state, cur_user1_action_index, cur_r_user1, next_user1_state)  # s, a, r, s_
            user2_qtable.learn(cur_user2_state, cur_user2_action_index, cur_r_user2, next_user2_state)
            mcs_qtable.learn(cur_mcs_state, cur_mcs_action_index, cur_r_mcs, next_mcs_state)

            # 6. record parameter
            # utility
            matrix_utility_mcs[episode][step] = cur_r_mcs
            matrix_utility_user1[episode][step] = cur_r_user1
            matrix_utility_user2[episode][step] = cur_r_user2
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
            pre_mcs_action, pre_mcs_action_index, pre_mcs_state = cur_mcs_action, cur_mcs_action_index, cur_mcs_state
            pre_user1_action, pre_user1_action_index, pre_user1_state = cur_user1_action, cur_user1_action_index, cur_user1_state
            pre_user2_action, pre_user2_action_index, pre_user2_state = cur_user2_action, cur_user2_action_index, cur_user2_state
    # 8. persistence parameter
    # single, data, utility, greedy, mcs, max_step
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
                       'q_learning',
                       'single',
                       f,
                       n_user)


def game_n_user(n_user=N_USER_MULTI, func=FUNC):  # todo 减少内存开销
    agent_mcs = MCSAgent(MCS_ACTION)
    agent_users = UserAgent(USER_ACTION, USER_COST, n_user)
    mcs_qtable = QLearningSingleUser(actions=agent_mcs.get_actions_index())
    user_qtable = QLearningMultiUser(n_user, actions=agent_users.get_actions_index())
    matrix_utility_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_action_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_action_mcs_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user_index = get_saved_matrix_mulit(MAX_EPISODE, MAX_STEP, n_user)

    matrix_aggregate_error = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    for episode in tqdm(range(MAX_EPISODE)):
        # clear q-table
        mcs_qtable.clear()
        user_qtable.clear_all()
        # init action
        pre_mcs_action_index, pre_mcs_action = agent_mcs.init_action_and_index_by_random()
        pre_user_action_index, pre_user_action = agent_users.multi_init_action_and_index_by_random()
        for step in range(MAX_STEP):
            # 1. composite state
            # 1.1 server state. number of actions selected by all clients for each action
            cur_user_action_num = agent_users.zero_actions_len()
            for idx in range(n_user):
                cur_user_action_num[int(pre_user_action_index[idx])] += 1
            cur_mcs_state = str(cur_user_action_num)
            # 1.2 client state. action of each client and server
            cur_user_state = agent_users.zero_user_state_len()
            for idx in range(n_user):
                cur_user_state[idx] = str([pre_mcs_action_index, pre_user_action_index[idx]])

            # 2. select action with q table
            # 2.1 server
            cur_mcs_action_index = mcs_qtable.select_action(cur_mcs_state, 'e-greedy')
            cur_mcs_action = agent_mcs.get_action_value(cur_mcs_action_index)
            # 2.2 client
            cur_user_action_index = agent_users.zero_user_len(tt=np.int32)
            cur_user_action = agent_users.zero_user_len(tt=np.float32)
            for idx in range(n_user):
                cur_user_action_index[idx] = user_qtable.select_action(int(idx), cur_user_state[idx], 'e-greedy')
                cur_user_action[idx] = agent_users.get_action_value(cur_user_action_index[idx])

            # 3. game
            # 3.1 leader multicast the payment list
            payment = agent_mcs.get_payments(cur_mcs_action,  agent_users.get_actions_len(), PAYMENT_ACC)

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

            cur_r_mcs = agent_mcs.get_mcs_reward(mcs_utility_all, cur_user_action_index, payment)

            # 3.2 followers get utility
            cur_r_user = agent_users.zero_user_len(tt=np.float32)
            for idx in range(n_user):
                cur_r_user[idx] = payment[cur_user_action_index[idx]] - agent_users.get_cost_value(cur_user_action_index[idx]) * cur_user_action[idx]

            # 4. get next state
            next_user_action_num = agent_users.zero_actions_len()
            for idx in range(n_user):
                next_user_action_num[cur_user_action_index[idx]] += 1

            # 4.1  server state. number of actions selected by all clients for each action
            next_mcs_state = str(next_user_action_num)
            # 4.2 client state. action of each client and server
            next_user_state = agent_users.zero_user_state_len()
            for idx in range(n_user):
                next_user_state[idx] = str([cur_mcs_action_index, cur_user_action_index[idx]])

            # 5. learning through q learning
            for idx in range(n_user):
                user_qtable.learn(idx, cur_user_state[idx], cur_user_action_index[idx], cur_r_user[idx], next_user_state[idx])  # s, a, r, s_
            mcs_qtable.learn(cur_mcs_state, cur_mcs_action_index, cur_r_mcs, next_mcs_state)

            # 6. record parameter
            # 6.1 client
            for idx in range(n_user):
                # utility
                matrix_utility_user[episode][step][idx] = cur_r_user[idx]
                # action
                matrix_action_user[episode][step][idx] = cur_user_action[idx]
                # action index
                matrix_action_user_index[episode][step][idx] = cur_user_action_index[idx]
            # 6.2 server
            matrix_action_mcs_index[episode][step] = cur_mcs_action_index
            matrix_action_mcs[episode][step] = cur_mcs_action
            matrix_utility_mcs[episode][step] = cur_r_mcs
            # aggregate error
            matrix_aggregate_error[episode][step] = aggregate_error

            # 7. update state
            pre_mcs_action, pre_mcs_action_index, pre_mcs_state = cur_mcs_action, cur_mcs_action_index, cur_mcs_state
            pre_user_action, pre_user_action_index, pre_user_state = cur_user_action, cur_user_action_index, cur_user_state

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
                      'q_learning',
                      'multi',
                      f,
                      n_user)


if __name__ == '__main__':
    game_2user(n_user=N_USER_SINGLE, func=FUNC)
    game_n_user(n_user=N_USER_MULTI, func=FUNC)
