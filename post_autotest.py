from __future__ import print_function
import os
import sys
import shutil
import glob
import pickle
import numpy as np
import pandas as pd
from monty.serialization import loadfn
from scipy import interpolate
from PIL import Image

from math import sqrt,pi,log,sin,cos,fabs
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
from os.path import expanduser
home_dir = expanduser("~")
sys.path.append(os.path.abspath(home_dir+'/template/python/src/matplotlib'))
from default_setup import mpl_default
sys.path.append(os.path.abspath(home_dir+'/template/python/src/util'))
from constants import eVtoJ, kB, hP, NA
from matplotlib.lines import Line2D

conf = sys.argv[1]
out_path = os.path.join(os.getcwd(), 'autotests/post')
eos_benchmark = '/home/zhuoyli/benchmarks/Mo/eos/bcc_dft'
gamma_benchmark = '/home/zhuoyli/benchmarks/Mo/gamma/110_14l'

class PlotFig:
  def __init__(self):
    #self.output_pic_fname = name
    self.fontsize = 18
    self.fig_grid = (1,1)
    self.fig_size = (8,6)
    self.mpld = mpl_default()
    self.mpld.setup(fontsize=self.fontsize)
    self.fig,self.ax = plt.subplots(nrows=self.fig_grid[0],
                                    ncols=self.fig_grid[1],
                                    figsize=self.fig_size)
    self.prop_cycle = plt.rcParams['axes.prop_cycle']
    #self.colors = self.prop_cycle.by_key()['color']
    self.colors = plt.get_cmap('Dark2').colors

    self.markers = ['s','o','H','v','d']
    self.markers_size = [7,8,8,7,8]

  def plot(self, lines: np.ndarray, pic_name='my_plot',
           step=100, xlb='xlabel', ylb='ylabel', title=None):
    ax = self.ax
    for ii in range(len(lines)):
        line = lines[ii]
        x = line[:, 0]
        y = line[:, 1]
        x_m = np.linspace(line[0, 0], line[-1, 0], step)
        Spline = interpolate.make_interp_spline(x, y)
        y_m = Spline(x_m)
        if ii == 0:
            ax.plot(x_m, y_m, color='black', linewidth=2,
                    zorder=1, label='DFT', alpha=0.8, marker=self.markers[ii],
                    ms=self.markers_size[ii], mec='black', mfc='white')
        else:
            ax.plot(x_m, y_m, color=self.colors[ii], linewidth=2,
                zorder=1, label=f'model_0{ii}', alpha=0.8, marker=self.markers[ii],
                ms=self.markers_size[ii], mec=self.colors[ii], mfc='white')

    ax.legend()
    if title:
        ax.set_title(title, fontsize=self.fontsize, fontweight='bold')
    ax.set_xlabel(xlb, fontsize=self.fontsize,fontweight='bold')
    ax.set_ylabel(ylb, fontsize=self.fontsize,fontweight='bold')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(pic_name, transparent=True, bbox_inches='tight')


def load_v(filename):
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r


def elastic(strategy_list):
    print('->> posting elastic results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        result_txt.append('\tC11  ' + '\tC12  ' + '\tC44  ' + '\n')
        result_table = []
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('elastic_00')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            r = np.loadtxt(result_orig, dtype=str, skiprows=1)
            result = [r[0, 0], r[0, 1], r[3, 3]]
            result_txt.append(f'\t{r[0, 0]}'f'\t{r[0, 1]}'f'\t{r[3, 3]}\n')
            result_table.append(result)
        os.chdir(strategy)
        result_txt.append('\n')
    out_file = os.path.join(out_path, 'elastic.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])
        """
        idx = []
        for ii in range(len(models)):
            idx.append(f'model_00{ii}')
        clm = ['C11', 'C12', 'C44']
        df = pd.DataFrame(result_table, index=idx, columns=clm)
        """


def relax(strategy_list):
    print('->> posting relaxation results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        result_txt.append('\ta(A)' + '\tEnergy(eV/atom)' + '\n')
        # result_table = []
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('relaxation/relax_task')
            result_json = os.path.join(os.getcwd(), 'result.json')
            r = loadfn(result_json)
            a = '%.3f' % r['cells'][1][0][0]
            if conf == 'std-bcc' or conf == 'std-hcp':
                energy = '%.3f' % (r['energies'][0]/2)
            elif conf == 'std-fcc':
                energy = '%.3f' % (r['energies'][0]/4)
            else:
                print(f'Error! illegal input configure: {conf}...')
                print('will exit...')
                exit()
            # result = [r[3]]
            result_txt.append(f'\t{a}\t{energy}\n')
            # result_table.append(result)
        os.chdir(strategy)
        result_txt.append('\n')
    out_file = os.path.join(out_path, 'relaxation.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def vacancy(strategy_list):
    print('->> posting vacancy results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        result_txt.append('\tVac_E(eV)' + '\n')
        # result_table = []
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('vacancy_00')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            r = np.loadtxt(result_orig, dtype=str, skiprows=2)
            # result = [r[3]]
            result_txt.append(f'\t{r[3]}\n')
            # result_table.append(result)
        os.chdir(strategy)
        result_txt.append('\n')
    out_file = os.path.join(out_path, 'vacancy.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def interstitial(strategy_list):
    print('->> posting interstitial results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        result_txt.append('\tTetrah' + '\tOctah' + '\tCrowd' +
                          '\t<111>db' + '\t<110>db' + '\t<100>db' + '\n')
        # result_table = []
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('interstitial_00')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            r = np.loadtxt(result_orig, dtype=str, skiprows=3)
            # result = [r[3]]
            result_txt.append(f'\t{r[0, 3]}\t{r[1, 3]}\t{r[2, 3]}\t{r[3, 3]}\t{r[4, 3]}\t{r[5, 3]}\n')
            # result_table.append(result)
        os.chdir(strategy)
        result_txt.append('\n')
    out_file = os.path.join(out_path, 'interstitial.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def surface(strategy_list):
    print('->> posting surface results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        result_txt.append('\t{110}' + '\t{100}' + '\t{111}' +
                          '\t{211}' + '\t{210}' + '\t{221}' +
                          '\t{311}' + '\t{321}' + '\t{320}' +
                          '\t{310}' + '\t{322}' + '\t{331}' +
                          '\t{332}' + '\n')
        # result_table = []
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('surface_00')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            r = np.loadtxt(result_orig, dtype=str, skiprows=2)
            # result = [r[3]]
            result_txt.append(f'\t{r[3, 3]}\t{r[9, 3]}\t{r[0, 3]}' +
                              f'\t{r[11, 3]}\t{r[12, 3]}\t{r[10, 3]}' +
                              f'\t{r[7, 3]}\t{r[5, 3]}\t{r[6, 3]}' +
                              f'\t{r[8, 3]}\t{r[4, 3]}\t{r[2, 3]}' +
                              f'\t{r[1, 3]}' + '\n')
            # result_table.append(result)
        os.chdir(strategy)
        result_txt.append('\n')
    out_file = os.path.join(out_path, 'surface.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def eos(strategy_list):
    print('->> posting EOS results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        #result_txt.append('\tVpA(A^3)' + '\tEpA(eV)' + '\n')
        result_table = []
        benchmark = r = np.loadtxt(eos_benchmark, dtype=float)
        result_table.append(benchmark)
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('eos_00')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            r = np.loadtxt(result_orig, dtype=float, skiprows=2)
            with open(result_orig, 'r') as f:
                result = f.readlines()[1:]
            for ii in range(len(result)):
                result_txt.append(result[ii])
            result_table.append(r)

        #os.chdir(strategy)
        result_txt.append('\n')
        plot_eos(data=result_table, name=strategy_name)
        #os.chdir(strategy)
    out_file = os.path.join(out_path, 'eos.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def plot_eos(data, name):
    os.chdir(out_path)
    if not os.path.isdir('eos_figures'):
        os.mkdir('eos_figures')
    os.chdir('eos_figures')
    pf = PlotFig()
    xlabel = 'Volume per atom (Å$^{3}$)'
    ylabel = r'Energy per atom ${E}$ (eV/atom)'
    pf.plot(data, pic_name=name, step=100,
            xlb=xlabel, ylb=ylabel, title=name)


def gamma(strategy_list):
    print('->> posting Gamma results <<-')
    result_txt = []
    strategy_list.sort()
    for strategy in strategy_list:
        os.chdir(strategy)
        strategy_name = strategy.split('/')[-1]
        result_txt.append(strategy_name + '\n')
        #result_txt.append('\tVpA(A^3)' + '\tEpA(eV)' + '\n')
        result_table = []
        benchmark = r = np.loadtxt(gamma_benchmark, dtype=float)
        result_table.append(benchmark)
        models = glob.glob(os.path.join(strategy, '00?'))
        models.sort()
        for model in models:
            print(f'working on {model}')
            os.chdir(model)
            os.chdir('confs')
            os.chdir(conf)
            os.chdir('gamma_00')
            result_txt.append('\tDisp \tStacking_Fault_E(J/m^2)\n')
            result_orig = os.path.join(os.getcwd(), 'result.out')
            with open('./result.out', 'r') as f:
                lines = f.readlines()
                result_list = []
                for line in lines[2:]:
                    data_dual = line.split()[3:5]
                    x, y = data_dual
                    data_list = [float(x), float(y)]
                    txt = '\t' + x + '\t' + y + '\n'
                    result_txt.append(txt)
                    result_list.append(data_list)
            result_table.append(np.array(result_list))
        # os.chdir(strategy)
        result_txt.append('\n')
        plot_gamma(data=result_table, name=strategy_name)
        # os.chdir(strategy)
    out_file = os.path.join(out_path, 'gamma.out')
    print(f'writing to {out_file}...')
    with open(out_file, 'w') as f:
        for ii in range(len(result_txt)):
            f.write(result_txt[ii])


def plot_gamma(data, name):
    os.chdir(out_path)
    if not os.path.isdir('gamma_figures'):
        os.mkdir('gamma_figures')
    os.chdir('gamma_figures')
    pf = PlotFig()
    xlabel = 'Fault displacement along 1/2[111]'
    ylabel = 'Fault energy ${E}$ (J/m$^{2}$)'
    pf.plot(data, pic_name=name, step=100,
            xlb=xlabel, ylb=ylabel, title=name)


def main(props, strategy_list):
    for prop in props:
        if prop == 'elastic':
            elastic(strategy_list)
        elif prop == 'relax':
            relax(strategy_list)
        elif prop == 'vacancy':
            vacancy(strategy_list)
        elif prop == 'interstitial':
            interstitial(strategy_list)
        elif prop == 'surface':
            surface(strategy_list)
        elif prop == 'eos':
            eos(strategy_list)
        elif prop == 'gamma':
            gamma(strategy_list)
        else:
            print(f'error! input unsupported property type: {prop}')
            print('will exit...')


if __name__ == '__main__':
    cwd = os.getcwd()
    if not os.path.exists('autotests'):
        print('Error! Missing working direction: autotests\n' +
              'please run autotest_helper.py first\n' +
              'will exist...')
        exit()

    if os.path.exists(out_path):
        print('post direction already exist...\n')
        is_rm = input('remove it?(y/n): ')
        if is_rm == 'y':
            shutil.rmtree(out_path)
            print('removed...')
        else:
            print('will exit...')
            exit()

    main_path = os.path.join(os.getcwd(), 'autotests')
    os.chdir(main_path)

    if os.path.exists(out_path):
        print('post direction already exist...\n')
        is_rm = input('remove it?(y/n): ')
        if is_rm == 'y':
            shutil.rmtree(out_path)
            print('removed...')
        else:
            print('will exit...')
            exit()
    else:
        print('creating post path...')
        os.mkdir('post')

    props = []
    while True:
        get_prop = input('please input property type for post:')

        if get_prop == 'all':
            props = ['relax', 'elastic', 'eos', 'surface',
                     'vacancy', 'interstitial', 'gamma']
            break
        elif get_prop == 'end':
            break
        else:
            props.append(get_prop)
            print(f'recorded: {get_prop}')

    try:
        strategy_list = load_v('strategy_list')
        model_list = load_v('model_list')
    except:
        print('missing path dump files: strategy_list and model_list\n' +
              'please follow steps of autotest_helper.py first\n' +
              'will exit...')
        exit()
    else:
        main(props, strategy_list)
        print('-<< finished >>-')
