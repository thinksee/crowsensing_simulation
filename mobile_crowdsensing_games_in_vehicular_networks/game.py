__author__ = 'think see'
__date__ = '2018/11/19'

from mobile_crowdsensing_games_in_vehicular_networks.model import *
from mobile_crowdsensing_games_in_vehicular_networks.qlearning import *
from mobile_crowdsensing_games_in_vehicular_networks.param import *
from mobile_crowdsensing_games_in_vehicular_networks.utils import *
from tqdm import tqdm

# create the document which can different the img and data.
if not os.path.exists('result\\img'):
    os.makedirs('result\\img')

if not os.path.exists('result\\data'):
    os.makedirs('result\\data')


def game():
    # init the parameter of the platform and user
    agent_mcs = MCSAgent(AGENT_MCS_ACTIONS, CONTRIBUTION_FACTOR)
    agent_user1 = UserAgent(V_ACTIONS,
                            SNR_SET,
                            SNR_PROB_SET,
                            V_GAMMA,
                            MAX_SPEED,
                            V_SPEED_SET,
                            V_SPEED_PROB_SET,
                            V_COST_SET)
    agent_user2 = UserAgent(V_ACTIONS,
                            SNR_SET,
                            SNR_PROB_SET,
                            V_GAMMA,
                            MAX_SPEED,
                            V_SPEED_SET,
                            V_SPEED_PROB_SET,
                            V_COST_SET)

    qlearning_mcs = QLearningTable(actions=agent_mcs.get_actions_index())  # 11*11 = 121
    qlearning_user1 = QLearningTable(actions=agent_user1.get_actions_index())  # 26*2 = 52
    qlearning_user2 = QLearningTable(actions=agent_user2.get_actions_index())  # 26*2 = 52

    matrix_utility_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user1 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_utility_user2 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_action_mcs = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user1 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user2 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_action_mcs_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user1_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_action_user2_index = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    matrix_cost_user1 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)
    matrix_cost_user2 = get_saved_matrix_single(MAX_EPISODE, MAX_STEP)

    for episode in tqdm(range(MAX_EPISODE)):
        qlearning_mcs.clear()
        qlearning_user1.clear()
        qlearning_user2.clear()
        # platform select a action .i.e the base of the payment,
        # user select a action and snr
        pre_mcs_action_index, pre_mcs_action = agent_mcs.init_action_and_index_by_random()
        pre_user1_action_index, pre_user1_action = agent_user1.init_action_and_index_by_random()
        pre_user2_action_index, pre_user2_action = agent_user2.init_action_and_index_by_random()

        pre_user1_snr_index, pre_user1_snr = agent_user1.init_snr_and_index_by_random()
        pre_user2_snr_index, pre_user2_snr = agent_user2.init_snr_and_index_by_random()

        pre_user1_speed = agent_user1.get_speed_value()
        pre_user2_speed = agent_user2.get_speed_value()

        for step in range(MAX_STEP):
            # 1. composite state
            cur_mcs_state = str([pre_user1_action_index, pre_user2_action_index])
            cur_user1_state = str([pre_mcs_action_index, pre_user1_snr_index])
            cur_user2_state = str([pre_mcs_action_index, pre_user2_snr_index])

            # 2. select action with q learning
            cur_mcs_action_index = qlearning_mcs.select_action(cur_mcs_state, POLICY)
            cur_mcs_action = agent_mcs.get_action_value(cur_mcs_action_index)
            cur_user1_action_index = qlearning_user1.select_action(cur_user1_state, POLICY)
            cur_user1_action = agent_user1.get_action_value(cur_user1_action_index)
            cur_user2_action_index = qlearning_user2.select_action(cur_user2_state, POLICY)
            cur_user2_action = agent_user2.get_action_value(cur_user2_action_index)

            cur_user1_cost = agent_user1.get_cost_value(cur_user1_action_index)
            cur_user2_cost = agent_user2.get_cost_value(cur_user2_action_index)

            cur_user1_snr, cur_snr_prob1 = agent_user1.get_snr_and_prob(pre_user1_speed, pre_user1_snr)
            cur_user2_snr, cur_snr_prob2 = agent_user2.get_snr_and_prob(pre_user2_speed, pre_user2_snr)

            cur_user1_snr_index = agent_user1.get_snr_index(cur_user1_snr)
            cur_user2_snr_index = agent_user2.get_snr_index(cur_user2_snr)

            payment = agent_mcs.get_payments(cur_mcs_action, agent_user1.get_action_length(), PAYMENT_ACC)

            # get the reward
            # y(xi) - xi * ci / hi
            cur_user1_reward = payment[cur_user1_action_index] - cur_user1_cost * cur_user1_action / cur_user1_snr
            cur_user2_reward = payment[cur_user2_action_index] - cur_user2_cost * cur_user2_action / cur_user2_snr
            # sum(beta(i) * xi - y(xi))
            cur_r_mcs = agent_mcs.get_utility_value(payment, cur_user1_action, cur_user1_action_index, cur_user2_action, cur_user2_action_index)

            # get next state
            next_mcs_state = str([cur_user1_action_index, cur_user2_action_index])
            next_user1_state = str([cur_mcs_action_index, cur_user1_snr_index])
            next_user2_state = str([cur_mcs_action_index, cur_user2_snr_index])
            # learning s, a, r, s_
            qlearning_mcs.learn(cur_mcs_state, cur_mcs_action_index, cur_r_mcs, next_mcs_state)
            qlearning_user1.learn(cur_user1_state, cur_user1_action_index, cur_user1_reward, next_user1_state)
            qlearning_user2.learn(cur_user2_state, cur_user2_action_index, cur_user2_reward, next_user2_state)

            # PDS-learning
            for i in range(agent_mcs.get_action_length()):
                for snr1 in range(agent_user1.get_snr_length()):
                    for action in range(agent_user1.get_action_length()):
                        temp = 0
                        for snr2 in range(agent_user1.get_snr_length()):
                            prob = agent_user1.select_prob_by_snr(cur_snr_prob1, snr1, snr2)
                            value = qlearning_user1.get_table_point_value(str([i, snr1]), action)
                            temp += prob * value
                        qlearning_user1.set_table_point_value(str([i, snr1]), action, temp)
                for snr1 in range(agent_user2.get_snr_length()):
                    for action in range(agent_user2.get_action_length()):
                        temp = 0
                        for snr2 in range(agent_user2.get_snr_length()):
                            prob = agent_user2.select_prob_by_snr(cur_snr_prob2, snr1, snr2)
                            value = qlearning_user2.get_table_point_value(str([i, snr1]), action)
                            temp += prob * value
                        qlearning_user2.set_table_point_value(str([i, snr1]), action, temp)

            matrix_utility_mcs[episode][step] = cur_r_mcs
            matrix_utility_user1[episode][step] = cur_user1_reward
            matrix_utility_user2[episode][step] = cur_user2_reward

            matrix_action_mcs[episode][step] = cur_mcs_action
            matrix_action_user1[episode][step] = cur_user1_action
            matrix_action_user2[episode][step] = cur_user2_action

            matrix_action_mcs_index[episode][step] = cur_mcs_action_index
            matrix_action_user1_index[episode][step] = cur_user1_action_index
            matrix_action_user2_index[episode][step] = cur_user2_action_index

            matrix_cost_user1[episode][step] = cur_user1_cost
            matrix_cost_user2[episode][step] = cur_user1_cost

            pre_mcs_action, pre_mcs_action_index = cur_mcs_action, cur_mcs_action_index
            pre_user1_action_index, pre_user1_snr_index = cur_user1_action_index, cur_user1_action_index
            pre_user2_action_index, pre_user2_snr_index = cur_user2_action_index, cur_user2_action_index

    save_to_txt_single(matrix_utility_mcs, 'utility', POLICY, 'mcs', MAX_STEP)
    save_to_txt_single(matrix_utility_user1, 'utility', POLICY, 'user1', MAX_STEP)
    save_to_txt_single(matrix_utility_user2, 'utility', POLICY, 'user2', MAX_STEP)

    save_to_txt_single(matrix_action_mcs, 'action', POLICY, 'mcs', MAX_STEP)
    save_to_txt_single(matrix_action_user1, 'action', POLICY, 'user1', MAX_STEP)
    save_to_txt_single(matrix_action_user2, 'action', POLICY, 'user2', MAX_STEP)

    save_to_txt_single(matrix_action_mcs_index, 'action-index', POLICY, 'mcs', MAX_STEP)
    save_to_txt_single(matrix_action_user1_index, 'action-index', POLICY, 'user1', MAX_STEP)
    save_to_txt_single(matrix_action_user2_index, 'action-index', POLICY, 'user2', MAX_STEP)

    save_to_txt_single(matrix_cost_user1, 'cost', POLICY, 'user1', MAX_STEP)
    save_to_txt_single(matrix_cost_user2, 'cost', POLICY, 'user2', MAX_STEP)

    plot_result_single(matrix_utility_mcs,
                       matrix_utility_user1,
                       matrix_utility_user2,
                       matrix_action_mcs,
                       matrix_action_user1,
                       matrix_action_user2,
                       matrix_action_mcs_index,
                       matrix_action_user1_index,
                       matrix_action_user2_index,
                       matrix_cost_user1,
                       matrix_cost_user2,
                       MAX_EPISODE,
                       MAX_STEP,
                       POLICY)


if __name__ == '__main__':
    game()
