__author__ = 'alibaba'
__date__ = '2019/5/24'

# reference  https://ieeexplore.ieee.org/document/8355763/
import numpy as np
import matplotlib.pyplot as plt
import os

MAX_EPISODE = 100
N_USER = 100
DATA_RANGE = 100
content = 'aggregate-error'
if not os.path.exists('img'):
    os.makedirs('img')

if not os.path.exists('data'):
    os.makedirs('data')


def get_aggregate_error(n_user, confidence_level, data_range, privacy_params_set):
    """
                              sqrt(2) * data_range
    aggregate-error error =  ------------------------------------ * sqrt (sum((pow(privacy_params, 2))[1, n_user])
                        n_user * sqrt(1 - confidence_level)

    """
    square_reciprocal_sum = 0
    for user in range(n_user):
        square_reciprocal_sum += (1 / np.square(np.random.choice(privacy_params_set, 1)))

    return np.sqrt(2) * data_range / (n_user * np.sqrt(1 - confidence_level)) * np.sqrt(square_reciprocal_sum)


def get_aggregate_error_infer(n_user, confidence_level, data_range, privacy_params_set):
    """
                                   sum((pow(privacy_params, 2))[1, n_user]
    aggregate-error error = data_range * -----------------------------------------
                                        n_user * (1 - confidence_level)
    """
    square_reciprocal_sum = 0
    for user in range(n_user):
        square_reciprocal_sum += (1 / np.square(np.random.choice(privacy_params_set)))

    return data_range * square_reciprocal_sum / (n_user * (1 - confidence_level))


def get_mcs_utility_reciprocal(n_user, confidence_level, data_range, privacy_params_set):
    """
    According to the formula, calculate the utility value before the server pays.
    :param n_user: the number of the user who participate in the data transmission.
    :param confidence_level: confidence Level of Probability Estimation.
    :param data_range: Valid range of data, such as 10, 100, 1000.
    :param privacy_params_set: Alternative privacy sets, (0, 1),
            where 0 represent the user don't participate, 1 represent the user don't add the privacy.

    :return: the aggregate error and utility of the platform.
    """
    privacy_parameter_square_reci_sum = 0
    for user_action in range(n_user):
        user_privacy_parameter = np.random.choice(privacy_params_set)
        # print('The privacy parameter of the user {} is {}.'.format(user_action, user_privacy_parameter))
        privacy_parameter_square_reci_sum += (1 / np.square(user_privacy_parameter))
    utility = 4000
    aggregate_error = np.sqrt(2) * data_range / (n_user * np.sqrt(1 - confidence_level)) *\
                      np.sqrt(privacy_parameter_square_reci_sum)
    utility = 1 / aggregate_error * utility
    # print('The aggregate error is {}, and the utility is {}'.format(aggregate_error, utility))
    return aggregate_error, utility


def get_mcs_utility_percentage(n_user, confidence_level, data_range, privacy_params_set):
    privacy_parameter_square_reci_sum = 0
    for user_action in range(n_user):
        privacy_parameter_square_reci_sum += (1 / np.square(np.random.choice(privacy_params_set)))
    utility = 500
    utility = (1 - (np.sqrt(2) * data_range / (n_user * np.sqrt(n_user - confidence_level)) *
                    np.sqrt(1 / privacy_parameter_square_reci_sum)) / data_range) * utility
    aggregate_error = np.sqrt(2) * data_range / (n_user * np.sqrt(n_user - confidence_level)) *\
                    np.sqrt(1 / privacy_parameter_square_reci_sum)
    return aggregate_error, utility


def get_optimized_points(points):
    return points


def plot_result(x, y, xlabel, ylabel, title, polyline_labels_list, polyline_color_list, scatter_marker_list, filename, is_saved=True):
    # 2.8 * 300 * 1.7 * 300
    # filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    plt.figure(num=None, figsize=(5.6, 3.4), dpi=300)
    plt.grid(linestyle='--')
    # font
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    # tuple : [list, list,..., list]
    if isinstance(y, tuple):
        for i, element in enumerate(y):
            plt.plot(x, element, polyline_color_list[i], linewidth=1, label=polyline_labels_list[i])
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
        plt.savefig(os.path.abspath(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img/' + content, filename + '.png')), dpi=300, bbox_inches='tight')

    plt.close()


if __name__ == '__main__':
    aggregate_error = []
    aggregate_error_infer = []
    # test
    for EPISODE in range(MAX_EPISODE):
        aggregate_error.append(get_aggregate_error(60, 0.9, 10, np.arange(0, .01, .001)))
        aggregate_error_infer.append(get_aggregate_error(60, 0.9, 10, np.arange(0., .01, .001)))
    plt.figure()
    plt.scatter(range(MAX_EPISODE), aggregate_error)
    # plt.plot()
    plt.scatter(range(MAX_EPISODE), aggregate_error_infer)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img/' + content, 'range-'+ content +'.png')))
    plt.close()

    for _ in range(20):
        get_mcs_utility_reciprocal(60, 0.95, 10, np.arange(0.1, 1.0, 0.1))
    # n_user -> aggregate-error error
    aggregate_error_infer.clear()
    aggregate_error.clear()
    wide = np.arange(1, N_USER, 1)
    for USER in wide:
        tmp_infer = []
        tmp = []
        for EPISODE in range(MAX_EPISODE):
            tmp.append(get_aggregate_error(USER, 0.9, 10, np.arange(0, 1.1, .2)))
            tmp_infer.append(get_aggregate_error_infer(USER, 0.9, 10, np.arange(0, 1.1, .2)))
        aggregate_error_infer.append(np.argmax(np.bincount(tmp_infer)))
        aggregate_error.append(np.argmax(np.bincount(tmp)))
        tmp_infer.clear()
        tmp.clear()

    plot_result(x=wide,
                y=(aggregate_error_infer, aggregate_error),
                xlabel='Number of User Participation',
                ylabel='Aggregate Error',
                title='',
                polyline_labels_list=['Infer', 'Not infer'],
                polyline_color_list=['royalblue', 'darkorange'],
                scatter_marker_list=['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'],
                filename='nuser-aggregate-error')
    aggregate_error_infer.clear()
    aggregate_error.clear()
    wide = np.arange(1, DATA_RANGE, 1)
    for DATA in wide:
        tmp_infer = []
        tmp = []
        for EPISODE in range(MAX_EPISODE):
            tmp.append(get_aggregate_error(2, 0.9, DATA, np.arange(0, 11, 1)))
            tmp_infer.append(get_aggregate_error_infer(2, 0.9, DATA, np.arange(0, 11, 1)))
        aggregate_error_infer.append(np.argmax(np.bincount(tmp_infer)))
        aggregate_error.append(np.argmax(np.bincount(tmp)))
        tmp_infer.clear()
        tmp.clear()

    plot_result(x=wide,
                y=(aggregate_error_infer, aggregate_error),
                xlabel='Data Range in Mobile Crowdsensing',
                ylabel='Aggregate Error',
                title='',
                polyline_labels_list=['Infer', 'Not infer'],
                polyline_color_list=['royalblue', 'darkorange'],
                scatter_marker_list=['o', '^', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'],
                filename='data-range-aggregate-error')

    aggregate_error.clear()
    aggregate_error_infer.clear()
    wide = [1, 10, 20, 30, 40, 50, 100, 200, 500, 1000]
    for DATA in wide:
        tmp_infer = []
        tmp = []
        for EPISODE in range(MAX_EPISODE):
            tmp.append(get_aggregate_error(2, 0.95, 10, np.arange(0, DATA, DATA/8)))
            tmp_infer.append(get_aggregate_error_infer(2, 0.95, 10, np.arange(0, DATA, DATA/8)))
        aggregate_error_infer.append(np.argmax(np.bincount(tmp_infer)))
        aggregate_error.append(np.argmax(np.bincount(tmp)))
        tmp_infer.clear()
        tmp.clear()

    plot_result(x=wide,
                y=aggregate_error,
                xlabel='Privacy Range in Mobile Crowdsensing',
                ylabel='Aggregate Error',
                title='',
                polyline_labels_list=['Infer', 'Not infer'],
                polyline_color_list=['royalblue', 'darkorange'],
                scatter_marker_list=['o', '^', 'v', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'],
                filename='privacy-range-aggregate-error')
