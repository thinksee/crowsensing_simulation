__author__ = 'think see'
__date__ = '2019/6/17'
"""
This package includes some helpful tools which deal the files, 
plot figures, key functions that is the utility of server including
the payment given the users, and you can add other functions which
is't relationship your model.
uniform provisions are as follows: 
number 1 -> the utility of the mcs and user end.
number 2 -> the action of the mcs and user end.
number 3 -> other parameter, in the program is aggregate error.
"""
import matplotlib.pyplot as plt
import os
import numpy as np


def get_saved_matrix_single(episode, step):
    return np.zeros([episode, step])


def get_saved_matrix_mulit(episode, step, n_user):
    return np.zeros([episode, step, n_user])


def save_to_txt_single(data, utility, greedy, mcs, max_step, n_user=120, function='reciprocal'):
    if utility == 'utility':
        np.savetxt('data\\single\\1-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    elif utility == 'action':
        np.savetxt('data\\single\\2-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    else:
        np.savetxt('data\\single\\3-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')


def save_to_txt_multi(data, utility, greedy, mcs, max_step, n_user=120, function='reciprocal'):
    if utility == 'utility':
        np.savetxt('data\\multi\\1-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    elif utility == 'action':
        np.savetxt('data\\multi\\2-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')
    else:
        np.savetxt('data\\multi\\3-{}-{}-{}-{}-{}-{}.txt'.format(utility, greedy, mcs, max_step, n_user, function),
                   data, fmt='%.2f')


def plot_result_single(matrix_utility_mcs,
                       matrix_utility_user1,
                       matrix_utility_user2,
                       matrix_aggregate_error,
                       matrix_action_mcs_index,
                       matrix_action_user1_index,
                       matrix_action_user2_index,
                       max_episode,
                       max_step,
                       way,
                       sm,
                       func,
                       n_user):
    array_utility_mcs = np.sum(matrix_utility_mcs, axis=0) / max_episode
    array_utility_mcs = array_utility_mcs.transpose()
    array_utility_user1 = np.sum(matrix_utility_user1, axis=0) / max_episode
    array_utility_user1 = array_utility_user1.transpose()
    array_utility_user2 = np.sum(matrix_utility_user2, axis=0) / max_episode
    array_utility_user2 = array_utility_user2.transpose()

    array_aggregate_error = np.sum(matrix_aggregate_error, axis=0) / max_episode
    array_aggregate_error = array_aggregate_error.transpose()

    # plot. the utility of users and mcs.
    # plt.figure(1)
    plt.subplot(221)
    plt.ylabel('Utility of the User')
    plt.plot(range(max_step), array_utility_user1)
    plt.subplot(222)
    plt.plot(range(max_step), array_utility_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the MCS')
    plt.plot(range(max_step), array_utility_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'single-utility-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
    plt.close()
    # plot. the aggregate error of mcs
    # plt.figure(2)
    plt.xlabel('Time Slot')
    plt.ylabel('Aggregate Error of MCS')
    plt.plot(range(max_step), array_aggregate_error)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'single-aggregate-error-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.close()
    # plot. the action by users and mcs.
    # In each time slot, get the action index with the largest number of occurrences.
    matrix_action_mcs_index = matrix_action_mcs_index.transpose()
    matrix_action_user1_index = matrix_action_user1_index.transpose()
    matrix_action_user2_index = matrix_action_user2_index.transpose()
    index_mcs = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_mcs_index)]
    index_user1 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user1_index)]
    index_user2 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user2_index)]
    # plt.figure(3)
    plt.plot(range(max_step), index_user1)
    plt.plot(range(max_step), index_user2)
    plt.plot(range(max_step), index_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'single-index-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
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
                      func,
                      n_user):
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
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'multi-utility-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
    plt.close()
    # plot. the aggregate error of mcs
    plt.figure(2)
    plt.xlabel('Time Slot')
    plt.ylabel('Aggregate Error of MCS')
    plt.plot(range(max_step), array_aggregate_error)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'multi-aggregate-error-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
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
            os.path.dirname(__file__)), '{}\\img\\{}'.format(way, sm), 'multi-index-e-greedy-{}-{}-{}.png'.format(max_step, n_user, func))))
    plt.tight_layout()
    plt.close()


def plot_result(MAX_EPISODE, MAX_STEP):
    matrix_utility_mcs = np.loadtxt('data/single-utility-e-greedy-mcs-7000.txt')
    matrix_utility_user1 = np.loadtxt('data/single-utility-e-greedy-user1-7000.txt')
    matrix_utility_user2 = np.loadtxt('data/single-utility-e-greedy-user2-7000-reciprocal.txt')
    matrix_aggregate_error = np.loadtxt('data/single-aggregate-error-e-greedy-mcs-7000.txt')
    matrix_action_mcs_index = np.loadtxt('data/single-action-e-greedy-mcs-7000-reciprocal.txt')
    matrix_action_user1_index = np.loadtxt('data/single-action-e-greedy-user1-7000-reciprocal.txt')
    matrix_action_user2_index = np.loadtxt('data/single-action-e-greedy-user2-7000-reciprocal.txt')

    array_utility_mcs = np.sum(matrix_utility_mcs, axis=0) / MAX_EPISODE
    array_utility_mcs = array_utility_mcs.T
    array_utility_user1 = np.sum(matrix_utility_user1, axis=0) / MAX_EPISODE
    array_utility_user1 = array_utility_user1.T
    array_utility_user2 = np.sum(matrix_utility_user2, axis=0) / MAX_EPISODE
    array_utility_user2 = array_utility_user2.T

    array_aggregate_error = np.sum(matrix_aggregate_error, axis=0) / MAX_EPISODE
    array_aggregate_error = array_aggregate_error.T

    # plot 1. the utility of user and server
    # plt.figure(1)
    plt.subplot(221)
    plt.ylabel('Utility of the User')
    plt.plot(range(MAX_STEP), array_utility_user1)
    plt.subplot(222)
    plt.plot(range(MAX_STEP), array_utility_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the MCS')
    plt.plot(range(MAX_STEP), array_utility_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'img', 'single-utility-e-greedy-{}.png'.format(MAX_STEP))))
    plt.tight_layout()
    plt.close()

    # plot 2. the aggregate error of server
    # plt.figure(2)
    plt.xlabel('Time Slot')
    plt.ylabel('Aggregate Error of MCS')
    plt.plot(range(MAX_STEP), array_aggregate_error)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'img', 'single-aggregate-error-e-greedy-{}.png'.format(MAX_STEP))))
    plt.close()
    # 画图 用户和服务器的动作选择
    # plot 3. user and server choose the actions
    # 基本思路就是对每一行求出现次数最多的值，然后表示当前time slot选择的值
    matrix_action_mcs_index = matrix_action_mcs_index.transpose()
    matrix_action_user1_index = matrix_action_user1_index.transpose()
    matrix_action_user2_index = matrix_action_user2_index.transpose()
    index_mcs = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_mcs_index)]
    index_user1 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user1_index)]
    index_user2 = [np.argmax(np.bincount(line)) for line in np.int32(matrix_action_user2_index)]
    # plt.figure(3)
    plt.plot(range(MAX_STEP), index_user1, label='user1')
    plt.plot(range(MAX_STEP), index_user2, label='user2')
    plt.plot(range(MAX_STEP), index_mcs, label='mcs')
    plt.xlabel('Time Slot')
    plt.ylabel('Each Agent\"s Action')
    plt.legend()
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'img', 'single-index-e-greedy-{}.png'.format(MAX_STEP))))
    plt.tight_layout()


def plot_result_statistical(x, y, xlabel, ylabel, title, polyline_labels_list, polyline_color_list, scatter_marker_list, filename, is_saved=True):
    # 2.8 * 300 * 1.7 * 300
    # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    plt.figure(num=None, figsize=(5.6, 3.4), dpi=300)
    plt.grid(linestyle='--')
    # font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    # tuple : (list, list,..., list)
    if isinstance(y, tuple):
        for i, element in enumerate(y):
            if polyline_labels_list is not None:
                plt.plot(x, element, polyline_color_list[i], linewidth=1, label=polyline_labels_list[i])
            if scatter_marker_list is not None:
                plt.scatter(x, element, c=polyline_color_list[i], s=5, marker=scatter_marker_list[i])
    elif isinstance(y, list):
        plt.plot(x, y, polyline_color_list, label=polyline_labels_list)
    else:
        print('type is\'t one to one')
        plt.close()
        exit(1)
    plt.xticks(fontproperties='Times New Roman', fontsize=12)
    plt.yticks(fontproperties='Times New Roman', fontsize=12)
    plt.legend(loc='best', prop={'family': 'Times New Roman', 'size': 12})
    if title != '':
        plt.title(title, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(ylabel, fontdict={'family': 'Times New Roman', 'size': 12})
    if is_saved:
        if 'image_category' not in locals() and 'image_category' not in globals():
            image_category = ''
        plt.savefig(os.path.abspath(
            os.path.join(os.path.abspath(
                os.path.dirname(__file__)), 'img/' + image_category, filename + '.png')), dpi=300, bbox_inches='tight')

    plt.close()


