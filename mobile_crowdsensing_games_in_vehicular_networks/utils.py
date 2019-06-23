import matplotlib.pyplot as plt
import os
import numpy as np


def get_saved_matrix_single(episode, step):
    return np.zeros([episode, step])


def get_saved_matrix_mulit(episode, step, n_user):
    return np.zeros([episode, step, n_user])


# matrix_utility_mcs, 'utility', POLICY, 'mcs', MAX_STEP
def save_to_txt_single(data, utility, policy, mcs, max_step):
    if utility == 'utility':
        np.savetxt('result\\data\\1-{}-{}-{}-{}.txt'.format(utility, policy, mcs, max_step),
                   data, fmt='%.2f')
    elif utility == 'action':
        np.savetxt('result\\data\\2-{}-{}-{}.txt'.format(utility, policy, mcs, max_step),
                   data, fmt='%.2f')
    elif utility == 'cost':
        np.savetxt('result\\data\\3-{}-{}-{}-{}.txt'.format(utility, policy, mcs, max_step),
                   data, fmt='%.2f')
    else:
        np.savetxt('result\\data\\4-{}-{}-{}-{}.txt'.format(utility, policy, mcs, max_step),
                   data, fmt='%.2f')


def save_to_txt_multi(data, utility, policy, mcs, max_step, n_user=120, function='reciprocal', algorithm=1):
    if utility == 'utility':
        np.savetxt('result\\data\\multi\\{}\\1-{}-{}-{}-{}-{}-{}.txt'.format(algorithm, utility, policy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    elif utility == 'action':
        np.savetxt('result\\data\\multi\\{}\\2-{}-{}-{}-{}-{}-{}.txt'.format(algorithm, utility, policy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    else:
        np.savetxt('result\\data\\multi\\{}\\3-{}-{}-{}-{}-{}-{}.txt'.format(algorithm, utility, policy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')


def plot_result_single(matrix_utility_mcs,
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
                       max_episode,
                       max_step,
                       policy):
    array_utility_mcs = np.sum(matrix_utility_mcs, axis=0) / max_episode
    array_utility_mcs = array_utility_mcs.transpose()
    array_utility_user1 = np.sum(matrix_utility_user1, axis=0) / max_episode
    array_utility_user1 = array_utility_user1.transpose()
    array_utility_user2 = np.sum(matrix_utility_user2, axis=0) / max_episode
    array_utility_user2 = array_utility_user2.transpose()

    array_action_mcs = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_mcs.T)]
    array_action_user1 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user1.T)]
    array_action_user2 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user2.T)]

    array_action_mcs_index = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_mcs_index.T)]
    array_action_user1_index = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user1_index.T)]
    array_action_user2_index = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user2_index.T)]

    array_cost_user1 = np.sum(matrix_cost_user1, axis=0) / max_episode
    array_cost_user1 = array_cost_user1.transpose()
    array_cost_user2 = np.sum(matrix_cost_user2, axis=0) / max_episode
    array_cost_user2 = array_cost_user2.transpose()

    # plot. the utility of users and mcs.
    plt.figure(1)
    plt.subplot(221)
    plt.ylabel('Utility of the User')
    plt.plot(range(max_step), array_utility_user1)
    plt.subplot(222)
    plt.plot(range(max_step), array_utility_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the MCS')
    plt.plot(range(max_step), array_utility_mcs)
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result\\img',
                                             'single-utility-{}-{}.png'.format(policy, max_step))))
    plt.close()

    # plot. the action of users and mcs.
    plt.figure(2)
    plt.subplot(221)
    plt.ylabel('Action of the User')
    plt.plot(range(max_step), array_action_user1)
    plt.subplot(222)
    plt.plot(range(max_step), array_action_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Action of the MCS')
    plt.plot(range(max_step), array_action_mcs)
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result\\img',
                                             'single-action-{}-{}.png'.format(policy, max_step))))
    plt.close()

    # plot. the action index of users and mcs.
    plt.figure(3)
    plt.subplot(221)
    plt.ylabel('Action index of the User')
    plt.plot(range(max_step), array_action_user1_index)
    plt.subplot(222)
    plt.plot(range(max_step), array_action_user2_index)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Action index of the MCS')
    plt.plot(range(max_step), array_action_mcs_index)
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result\\img',
                                             'single-action-index-{}-{}.png'.format(policy, max_step))))
    plt.close()

    # plot. the action index of users and mcs.
    plt.figure(4)
    plt.subplot(211)
    plt.ylabel('Cost of the User1')
    plt.plot(range(max_step), array_cost_user1)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Cost of the User2')
    plt.plot(range(max_step), array_cost_user2)
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'result\\img',
                                             'single-cost-{}-{}.png'.format(policy, max_step))))
    plt.close()


def plot_result_multi(matrix_utility_mcs,
                      matrix_utility_user,
                      matrix_aggregate_error,
                      matrix_action_mcs_index,
                      matrix_action_user_index,
                      max_episode,
                      max_step,
                      way,
                      sm,
                      n_user,
                      func,
                      algorithm):
    array_utility_mcs = np.sum(matrix_utility_mcs, axis=0) / max_episode
    array_utility_mcs = array_utility_mcs.transpose()
    array_utility_user = np.sum(matrix_utility_user, axis=0) / max_episode
    array_utility_user = array_utility_user.transpose()
    array_aggregate_error = np.sum(matrix_aggregate_error, axis=0) / max_episode
    array_aggregate_error = array_aggregate_error.transpose()

    # plot. the utility of users and mcs.
    plt.figure(1)
    plt.subplot(211)
    plt.ylabel('Utility of the User')
    plt.plot(range(max_step), array_utility_user)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the MCS')
    plt.plot(range(max_step), array_utility_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'result\\img\\{}\\{}'.format(sm, algorithm),
            'multi-utility-e-policy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
    plt.close()
    # plot. the aggregate error of mcs
    plt.figure(2)
    plt.xlabel('Time Slot')
    plt.ylabel('Aggregate Error of MCS')
    plt.plot(range(max_step), array_aggregate_error)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'result\\img\\{}\\{}'.format(sm, algorithm),
            'multi-aggregate-error-e-policy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.close()
    # plot. the action by users and mcs.
    # In each time slot, get the action index with the largest number of occurrences.
    matrix_action_mcs_index = matrix_action_mcs_index.transpose()
    matrix_action_user_index = matrix_action_user_index.transpose()
    index_mcs = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_mcs_index)]
    index_user = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user_index)]
    plt.figure(3)
    plt.plot(range(max_step), index_user)
    plt.plot(range(max_step), index_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'result\\img\\{}\\{}'.format(sm, algorithm).format(way, sm),
            'multi-index-e-policy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
    plt.close()
