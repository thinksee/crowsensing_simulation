__author__ = 'alibaba'
__date__ = '2019/1/23'

import numpy as np
import matplotlib.pyplot as plt

pdf_name_list = [
    'action_best.pdf',
    'action_fake.pdf',
    'action_silence.pdf',
    'server_price.pdf',
    'utility_server.pdf',
    'utility_user.pdf',
    'action_bfs.pdf',
    'utility_su.pdf'
]
txt_name_list = [
    'action_bestlam_60_100.txt',
    'action_fakelam_60_100.txt',
    'action_silencelam_60_100.txt',
    'server_pricelam_60_100.txt',
    'Utility_serverlam_60_100.txt',
    'Utility_userlam_60_100.txt'
]


def text2list(file):
    with open(file, 'r', encoding='utf-8') as text:
       result = np.loadtxt(file,)

    return result


def list2pdf(result, pdf_name):
    plt.figure(0)
    plt.plot(result)
    plt.xlabel('Time slot')
    plt.ylabel('Utility')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name, format='pdf')
    plt.clf()


def dlist2pdf(result1, result2, pdf_name):
    plt.figure(0)
    plt.plot(result1)
    plt.plot(result2)
    plt.xlabel('Time slot')
    plt.ylabel('Utility')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name, format='pdf')
    plt.clf()


def tlist2pdf(result1, result2, result3, pdf_name):
    plt.figure(0)
    plt.plot(result1)
    plt.plot(result2)
    plt.plot(result3)
    plt.xlabel('Time slot')
    plt.ylabel('Utility')
    # plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name, format='pdf')
    plt.clf()


if __name__ == '__main__':
    for txt_name, pdf_name in zip(txt_name_list, pdf_name_list):
        list2pdf(text2list(txt_name), pdf_name)
    # for pdf_name in pdf_name_list:
    tlist2pdf(text2list(txt_name_list[0]), text2list(txt_name_list[1]), text2list(txt_name_list[2]), pdf_name_list[6])
    dlist2pdf(text2list(txt_name_list[4]), text2list(txt_name_list[5]), pdf_name_list[7])

