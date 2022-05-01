import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import scipy.optimize as opt
from scipy.optimize import curve_fit


def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst


def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr

def __sse(y_fitting, y_no_fitting):
    """
    计算SSE(error sum of squares) 残差平方和
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 残差平方和SSE
    """
    s_list = [(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))]
    sse = sum(s_list)
    return sse


def goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    rr = SSR /SST
    return rr


def uncentered_goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度 uncentered R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: uncentered 拟合优度R^2
    """
    SSE = __sse(y_fitting, y_no_fitting)
    sum_y_no_fit = np.dot(np.array(y_no_fitting), np.array(y_no_fitting))
    rr = 1 - SSE / sum_y_no_fit
    return rr


def expon_fit(x, a, b, c):
    return a * np.exp(-b * x) + c


# def power_fit(x, m, c, c0):
#     return c0 + (x**m) * c


def power_fit(x, m, c,):
    return (x**m) * c


def lognorm_fit(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))


def lognorm_fit_scale(x, mu, sigma, c):
    return (c / (x * sigma * np.sqrt(2 * np.pi))) * \
           np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))


# def lognorm_fit(x, mu, sigma):
#     return 1 / (x * sigma) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))


def main(data):
    x = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    # f = plt.figure(figsize=(12, 12), dpi=600)
    # ax1 = f.add_subplot(1,1,1)
    plt.scatter(x, y, color='r', s=5)
    candidate_fit = ['power law', 'exponential', 'log normal']
    candidate_popt = ['c0 + (x**m) * c', 'a * np.exp(-b * x) + c',
                      '(1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))']
    fit_r_r = []
    fit_popt = []

    popt, pcov = curve_fit(power_fit, np.array(x), np.array(y), maxfev=5000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, power_fit(x, *popt), linestyle='--', color='b', linewidth=2, label='power law')
    print(goodness_of_fit(power_fit(x, *popt), y))
    fit_r_r.append(goodness_of_fit(power_fit(x, *popt), y))

    popt, pcov = curve_fit(expon_fit, x / 10, np.array(y), maxfev=5000)
    # print(popt)
    fit_popt.append(popt)
    print(goodness_of_fit(expon_fit(x / 10, *popt), y))
    fit_r_r.append(goodness_of_fit(expon_fit(x / 10, *popt), y))
    ax1 = plt.plot(x, expon_fit(x / 10, *popt), linestyle=':', color='g', linewidth=2, label='exponential')

    popt, pcov = curve_fit(lognorm_fit, np.array(x / 100), np.array(y), maxfev=50000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, lognorm_fit(x / 100, *popt), linestyle='-.', color='y', label='log normal')
    print(goodness_of_fit(lognorm_fit(x / 100, *popt), y))
    fit_r_r.append(goodness_of_fit(lognorm_fit(x / 100, *popt), y))
    plt.legend(loc='upper right', fontsize=12)
    # 需要输出的
    print(candidate_fit[fit_r_r.index(max(fit_r_r))])
    print(candidate_popt[fit_r_r.index(max(fit_r_r))])
    print(fit_popt[fit_r_r.index(max(fit_r_r))])
    # print(popt)
    plt.text(x=5, y=0, s=str(candidate_fit[fit_r_r.index(max(fit_r_r))]),
             fontdict=dict(fontsize=12, color='r', family='monospace', ))
    plt.show()


def main_xy_with_scale(x, y, fig_path, xlabel='距离 （Km）', ylabel='单位到访人次 （人/km2）', log_swith='off'):
    x = np.array(x)
    y = np.array(y)
    scl = max(y) / 100
    log_scl = max(y)

    # f = plt.figure(figsize=(12, 12), dpi=600)
    # ax1 = f.add_subplot(1,1,1)
    plt.scatter(x, y, color='r', s=5)
    candidate_fit = ['power law', 'exponential', 'log normal']
    candidate_popt = ['(x**m) * c * scl', '(a * np.exp(-b * x) + c) * scl',
                      '((c / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))) * scl']
    fit_r_r = []
    fit_popt = []

    popt, pcov = curve_fit(power_fit, np.array(x), np.array(y)/scl, maxfev=5000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, power_fit(x, *popt)*scl, linestyle='--', color='b', linewidth=2, label='power law')
    print(uncentered_goodness_of_fit(power_fit(x, *popt), y/scl))
    fit_r_r.append(uncentered_goodness_of_fit(power_fit(x, *popt), y/scl))

    popt, pcov = curve_fit(expon_fit, x , np.array(y)/scl, maxfev=5000)
    # print(popt)
    fit_popt.append(popt)
    print(uncentered_goodness_of_fit(expon_fit(x, *popt), y/scl))
    fit_r_r.append(uncentered_goodness_of_fit(expon_fit(x, *popt), y/scl))
    ax1 = plt.plot(x, expon_fit(x, *popt)*scl, linestyle=':', color='g', linewidth=2, label='exponential')

    popt, pcov = curve_fit(lognorm_fit_scale, np.array(x), np.array(y/log_scl), maxfev=500000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, lognorm_fit_scale(x, *popt)*log_scl, linestyle='-.', color='y', label='log normal')
    print(uncentered_goodness_of_fit(lognorm_fit_scale(x, *popt), y/log_scl))
    fit_r_r.append(uncentered_goodness_of_fit(lognorm_fit_scale(x, *popt), y/log_scl))
    plt.legend(loc='upper right', fontsize=12)
    # 需要输出的
    print(candidate_fit[fit_r_r.index(max(fit_r_r))])
    print(candidate_popt[fit_r_r.index(max(fit_r_r))])
    print(fit_popt[fit_r_r.index(max(fit_r_r))])
    # print(popt)
    plt.title(str(candidate_fit[fit_r_r.index(max(fit_r_r))]))
    # plt.text(x=5, y=0, s=str(candidate_fit[fit_r_r.index(max(fit_r_r))]),
    #          fontdict=dict(fontsize=12, color='r', family='monospace', ))
    if log_swith == 'on':
        plt.xscale('log')
        plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fig_path)
    plt.close()
    # 增加缩放输出正确的公式
    if fit_r_r.index(max(fit_r_r)) == 2:
        scl_flag = log_scl
    else:
        scl_flag = scl
    return [candidate_fit[fit_r_r.index(max(fit_r_r))],
            candidate_popt[fit_r_r.index(max(fit_r_r))],
            np.append(fit_popt[fit_r_r.index(max(fit_r_r))], scl_flag),
            max(fit_r_r)]


def main_xy(x, y):
    x = np.array(x)
    y = np.array(y)
    # f = plt.figure(figsize=(12, 12), dpi=600)
    # ax1 = f.add_subplot(1,1,1)
    plt.scatter(x, y, color='r', s=5)
    candidate_fit = ['power law', 'exponential', 'log normal']
    candidate_popt = ['c0 + (x**m) * c', 'a * np.exp(-b * x) + c',
                      '(1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))']
    fit_r_r = []
    fit_popt = []

    popt, pcov = curve_fit(power_fit, np.array(x), np.array(y), maxfev=5000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, power_fit(x, *popt), linestyle='--', color='b', linewidth=2, label='power law')
    print(goodness_of_fit(power_fit(x, *popt), y))
    fit_r_r.append(goodness_of_fit(power_fit(x, *popt), y))

    popt, pcov = curve_fit(expon_fit, x / 10, np.array(y), maxfev=5000)
    # print(popt)
    fit_popt.append(popt)
    print(goodness_of_fit(expon_fit(x / 10, *popt), y))
    fit_r_r.append(goodness_of_fit(expon_fit(x / 10, *popt), y))
    ax1 = plt.plot(x, expon_fit(x / 10, *popt), linestyle=':', color='g', linewidth=2, label='exponential')

    popt, pcov = curve_fit(lognorm_fit, np.array(x), np.array(y/10000), maxfev=50000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, lognorm_fit(x, *popt)*10000, linestyle='-.', color='y', label='log normal')
    print(goodness_of_fit(lognorm_fit(x, *popt), y/10000))
    fit_r_r.append(goodness_of_fit(lognorm_fit(x, *popt), y/10000))
    plt.legend(loc='upper right', fontsize=12)
    # 需要输出的
    print(candidate_fit[fit_r_r.index(max(fit_r_r))])
    print(candidate_popt[fit_r_r.index(max(fit_r_r))])
    print(fit_popt[fit_r_r.index(max(fit_r_r))])
    # print(popt)
    plt.text(x=5, y=0, s=str(candidate_fit[fit_r_r.index(max(fit_r_r))]),
             fontdict=dict(fontsize=12, color='r', family='monospace', ))
    plt.show()


if __name__ == "__main__":
    data = pd.read_excel(r'C:\Users\29420\Documents\WeChat Files\wxid_avb0egdv9lo422\FileStorage\File\2022-03\data_test.xlsx', sheet_name='公园4')
    x = np.array(data.iloc[:,0])
    y = np.array(data.iloc[:,1])
    # f = plt.figure(figsize=(12, 12), dpi=600)
    # ax1 = f.add_subplot(1,1,1)
    plt.scatter(x, y, color='r', s=5)
    candidate_fit = ['power law', 'exponential', 'log normal']
    candidate_popt = ['c0 + (x**m) * c', 'a * np.exp(-b * x) + c', '(1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))']
    fit_r_r = []
    fit_popt = []

    popt, pcov = curve_fit(power_fit, np.array(x), np.array(y), maxfev=5000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, power_fit(x, *popt), linestyle='--', color='b', linewidth=2, label='power law')
    print(goodness_of_fit(power_fit(x, *popt), y))
    fit_r_r.append(goodness_of_fit(power_fit(x, *popt), y))

    popt, pcov = curve_fit(expon_fit, x/10, np.array(y), maxfev=5000)
    # print(popt)
    fit_popt.append(popt)
    print(goodness_of_fit(expon_fit(x/10, *popt), y))
    fit_r_r.append(goodness_of_fit(expon_fit(x/10, *popt), y))
    ax1 = plt.plot(x, expon_fit(x/10, *popt), linestyle=':', color='g', linewidth=2, label='exponential')


    popt, pcov = curve_fit(lognorm_fit, np.array(x/100), np.array(y), maxfev=50000)
    fit_popt.append(popt)
    ax1 = plt.plot(x, lognorm_fit(x/100, *popt), linestyle='-.', color='y', label='log normal')
    print(goodness_of_fit(lognorm_fit(x/100, *popt), y))
    fit_r_r.append(goodness_of_fit(lognorm_fit(x/100, *popt), y))
    plt.legend(loc='upper right', fontsize=12)
    # 需要输出的
    print(candidate_fit[fit_r_r.index(max(fit_r_r))])
    print(candidate_popt[fit_r_r.index(max(fit_r_r))])
    print(fit_popt[fit_r_r.index(max(fit_r_r))])
    # print(popt)
    plt.text(x=5, y=0, s=str(candidate_fit[fit_r_r.index(max(fit_r_r))]),
             fontdict=dict(fontsize=12, color='r',family='monospace',))
    plt.show()


