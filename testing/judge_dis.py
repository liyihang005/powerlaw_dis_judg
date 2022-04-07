from powerlaw import trim_to_range
import pylab
import urllib.request as ur
pylab.rcParams['xtick.major.pad']='8'
pylab.rcParams['ytick.major.pad']='8'
from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
rc('font', family='sans-serif')
rc('font', size=10.0)
rc('text', usetex=False)
from matplotlib.font_manager import FontProperties

panel_label_font = FontProperties().copy()
panel_label_font.set_weight("bold")
panel_label_font.set_size(12.0)
panel_label_font.set_family("sans-serif")
from os import listdir
files = listdir('.')
if 'blackouts.txt' not in files:

    ur.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/blackouts.txt', 'blackouts.txt')
if 'words.txt' not in files:

    ur.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/words.txt', 'words.txt')
if 'worm.txt' not in files:

    ur.urlretrieve('https://raw.github.com/jeffalstott/powerlaw/master/manuscript/worm.txt', 'worm.txt')

from numpy import genfromtxt
blackouts = genfromtxt('blackouts.txt')#/10**3
words = genfromtxt('words.txt')
worm = genfromtxt('worm.txt')
worm = worm[worm>0]


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


from numpy import unique


def get_fitted(obj):
    bins = unique(trim_to_range(obj.parent_Fit.data, xmin=obj.xmin, xmax=obj.xmax))
    return obj.pdf(bins)


def plot_basics(data, data_inst, fig, units):
    from powerlaw import plot_pdf, Fit, pdf
    annotate_coord = (-.4, .95)
    ax1 = fig.add_subplot(n_graphs, n_data, data_inst)
    x, y = pdf(data, linear_bins=True)
    ind = y > 0
    y = y[ind]
    x = x[:-1]
    x = x[ind]

    ax1.scatter(x, y, color='r', s=.5)
    plot_pdf(data[data > 0], ax=ax1, color='b', linewidth=0)
    # 第一个图的横轴刻度不显示
    # from pylab import setp
    # setp(ax1.get_xticklabels(), visible=True)

    if data_inst == 1:
        ax1.annotate("A", annotate_coord, xycoords="axes fraction", fontproperties=panel_label_font)
    #直方图
    # from mpl_toolkits.axes_grid.inset_locator import inset_axes
    # ax1in = inset_axes(ax1, width="30%", height="30%", loc=3)
    # ax1in.hist(data, density=True, color='b')
    # ax1in.set_xticks([])
    # ax1in.set_yticks([])

    ax2 = fig.add_subplot(n_graphs, n_data, n_data * 2 + data_inst, sharex=ax1)
    ax2.scatter(x, y, color='r', s=.5)
    # plot_pdf(data, ax=ax2, color='b', linewidth=2)
    # 按照最小值从1开始,
    # fit = Fit(data, xmin=1, discrete=True)
    # fit.power_law.plot_pdf(ax=ax2, linestyle=':', color='g')
    # p = fit.power_law.pdf()

    ax2.set_xlim(ax1.get_xlim())
    # 按照最小值从x的最小值开始,以后以此为标准来拟合
    fit = Fit(data, xmin=min(x), discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle='--', color='b')
    # from pylab import setp
    # setp(ax2.get_xticklabels(), visible=False)

    # if data_inst == 1:
    #     ax2.annotate("B", annotate_coord, xycoords="axes fraction", fontproperties=panel_label_font)
    #     ax2.set_ylabel(u"p(X)")  # (10^n)")

    '''
    bins = unique(trim_to_range(data, xmin=self.xmin, xmax=self.xmax))
    PDF = self.pdf(bins)
    '''

    fit = Fit(data, xmin=min(x), discrete=True)
    ax3 = fig.add_subplot(n_graphs, n_data, n_data + data_inst)  # , sharex=ax1)#, sharey=ax2)
    ax3.scatter(x, y, color='r', s=.5)
    fit.power_law.plot_pdf(ax=ax3, linestyle='--', color='b')
    fit.exponential.plot_pdf(ax=ax3, linestyle='--', color='c')
    fit.lognormal.plot_pdf(ax=ax3, linestyle='--', color='y')

    #TODO 检查一下能不能输出R方
    rr_pl = goodness_of_fit(get_fitted(fit.power_law), y)
    rr_ex = goodness_of_fit(get_fitted(fit.exponential), y)
    rr_lg = goodness_of_fit(get_fitted(fit.lognormal), y)
    # fit.plot_pdf(ax=ax3, color='b', linewidth=2)

    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax1.get_xlim())
    # ax3.set_ylim((1e-6, 1))
    # ax3.set_xlim((1, 1e5))

    if data_inst == 1:
        ax3.annotate("C", annotate_coord, xycoords="axes fraction", fontproperties=panel_label_font)

    ax3.set_xlabel(units)


if __name__ == "__main__":
    n_data = 3
    n_graphs = 4
    f = plt.figure(figsize=(12, 12))

    data = words
    data_inst = 1
    units = 'Word Frequency'
    plot_basics(data, data_inst, f, units)
    plt.show()

