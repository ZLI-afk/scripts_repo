import os
import shutil
import re
import sys
import glob
from scipy import interpolate
from PIL import Image
import numpy as np
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

CS: int = 8  # supercell size
CONF: str = 'bcc'

# benchmarks path
LATT = '/home/zhuoyli/benchmarks/Mo/lattice_finite_T/lattice'
C11 = '/home/zhuoyli/benchmarks/Mo/elastic_finite_T/c11'
C12 = '/home/zhuoyli/benchmarks/Mo/elastic_finite_T/c12'
C44 = '/home/zhuoyli/benchmarks/Mo/elastic_finite_T/c44'


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
           step=100, xlb='xlabel', ylb='ylabel', title=None, ref_label='DFT'):
    ax = self.ax
    for ii in range(len(lines)):
        line = lines[ii]
        x = line[:, 0]
        y = line[:, 1]
        x_m = np.linspace(line[0, 0], line[-1, 0], step)
        Spline = interpolate.make_interp_spline(x, y)
        y_m = Spline(x_m)
        if ii == 0:
            ax.plot(x, y, color='black', linewidth=2,
                    zorder=1, label=ref_label, alpha=0.8, marker=self.markers[ii],
                    ms=self.markers_size[ii], mec='black', mfc='white')
        else:
            ax.plot(x, y, color=self.colors[ii], linewidth=2,
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


def lattice(pot_paths):
    main_path = os.getcwd()
    if os.path.isdir('lattice_finite_T'):
        shutil.rmtree('lattice_finite_T')
    os.mkdir('lattice_finite_T')
    result_dir = os.path.join(main_path, 'lattice_finite_T')
    for pot in pot_paths:
        pot_name = pot.split('/')[-1]
        print(f'working on {pot_name}')
        os.chdir(pot)
        os.chdir(f'lat_param_finite_t/isothermal/{CS}_{CS}_{CS}/Mo')
        cwd = os.getcwd()
        temp_paths = glob.glob(os.path.join(cwd, f'{CONF}_*'))
        temp_paths.sort()
        result = []
        data = []
        for temp_path in temp_paths:
            dir_name = temp_path.split('/')[-1]
            print(f'    working on {dir_name}')
            temp = int(re.findall(f'{CONF}_(\d+)K', dir_name)[0])
            os.chdir(temp_path)
            log = glob.glob(os.path.join(os.getcwd(), 'in.*.log'))[0]
            with open(log, 'r', errors='ignore') as f1:
                content = f1.readlines()
                content_tail = ''.join(content[-100:])
                a = float(re.findall(f'variable lat_{CONF}_a_Mo equal (\d.\d+)', content_tail)[0])
                b = float(re.findall(f'variable lat_{CONF}_b_Mo equal (\d.\d+)', content_tail)[0])
                c = float(re.findall(f'variable lat_{CONF}_c_Mo equal (\d.\d+)', content_tail)[0])
                lat_param_ave = (a+b+c)/3
            result.append(f'{temp} {lat_param_ave}\n')
            data.append([temp, lat_param_ave])
        result_out = os.path.join(result_dir, f'{pot_name}.out')
        result_table = []
        benchmarks = np.loadtxt(LATT, dtype=float)
        result_table.append(benchmarks)
        table = np.array(data)
        table = table[np.argsort(table[:, 0]), :]
        result_table.append(table)
        plot_lattice(data=result_table, name=pot_name, out_path=result_dir)
        with open(result_out, 'w') as f2:
            for ii in result_table:
                txt = f'{ii[0]} {ii[1]}'
                f2.write(txt)


def plot_lattice(data, name, out_path):
    os.chdir(out_path)
    pf = PlotFig()
    xlabel = 'Temperature (K)'
    ylabel = 'Lattice parameter (Å)'
    pf.plot(data, pic_name=name, step=100,
            xlb=xlabel, ylb=ylabel, title=name, ref_label='DFT (Y Lysogorskiy et al 2019)')

def elastic(pot_paths):
    main_path = os.getcwd()
    if os.path.isdir('elastic_finite_T'):
        shutil.rmtree('elastic_finite_T')
    os.mkdir('elastic_finite_T')
    result_dir = os.path.join(main_path, 'elastic_finite_T')
    os.chdir(result_dir)
    os.mkdir('c11')
    os.mkdir('c12')
    os.mkdir('c44')
    for pot in pot_paths:
        pot_name = pot.split('/')[-1]
        print(f'working on {pot_name}')
        os.chdir(pot)
        os.chdir(f'elastic_tensor_finite_t/isothermal/{CS}_{CS}_{CS}/Mo')
        cwd = os.getcwd()
        temp_paths = glob.glob(os.path.join(cwd, f'{CONF}_*'))
        temp_paths.sort()
        result = []
        data = []
        for temp_path in temp_paths:
            dir_name = temp_path.split('/')[-1]
            print(f'    working on {dir_name}')
            temp = int(re.findall(f'{CONF}_(\d+)K', dir_name)[0])
            os.chdir(temp_path)
            log = glob.glob(os.path.join(os.getcwd(), 'in.*.log'))[0]
            with open(log, 'r', errors='ignore') as f1:
                content = f1.readlines()
                content_tail = ''.join(content[-300:])
                try:
                    c11 = float(re.findall(f'mod\nvariable {CONF}_Mo_C11 equal (\d+.\d+)', content_tail)[0])
                    c12 = float(re.findall(f'mod\nvariable {CONF}_Mo_C12 equal (\d+.\d+)', content_tail)[0])
                    c44 = float(re.findall(f'mod\nvariable {CONF}_Mo_C44 equal (\d+.\d+)', content_tail)[0])
                except IndexError:
                    c11 = float(re.findall(f'Elastic Constant C11all = (\d+.\d+) GPa', content_tail)[0])
                    c12 = float(re.findall(f'Elastic Constant C12all = (\d+.\d+) GPa', content_tail)[0])
                    c44 = float(re.findall(f'Elastic Constant C44all = (\d+.\d+) GPa', content_tail)[0])
            result.append(f'{temp} {c11} {c12} {c44}\n')
            data.append([temp, c11, c12, c44])
        result_out = os.path.join(result_dir, f'{pot_name}.out')
        result_table = np.array(data)
        result_table = result_table[np.argsort(result_table[:, 0]), :]
        c11_ta = result_table[:, [0, 1]]
        c12_ta = result_table[:, [0, 2]]
        c44_ta = result_table[:, [0, 3]]
        c11_label = 'Elastic constant C11 (GPa)'
        c12_label = 'Elastic constant C12 (GPa)'
        c44_label = 'Elastic constant C44 (GPa)'
        c11_table = []
        c12_table = []
        c44_table = []
        c11_table.append(np.loadtxt(C11, dtype=float))
        c12_table.append(np.loadtxt(C12, dtype=float))
        c44_table.append(np.loadtxt(C44, dtype=float))
        c11_table.append(c11_ta)
        c12_table.append(c12_ta)
        c44_table.append(c44_ta)
        plot_elastic(data=c11_table, name=pot_name,
                     out_path=result_dir+'/c11', ylabel=c11_label)
        plot_elastic(data=c12_table, name=pot_name,
                     out_path=result_dir + '/c12', ylabel=c12_label)
        plot_elastic(data=c44_table, name=pot_name,
                     out_path=result_dir + '/c44', ylabel=c44_label)
        with open(result_out, 'w') as f2:
            for ii in result_table:
                txt = f'{ii[0]} {ii[1]}'
                f2.write(txt)


def plot_elastic(data, name, out_path, ylabel):
    os.chdir(out_path)
    pf = PlotFig()
    xlabel = 'Temperature (K)'
    pf.plot(data, pic_name=name, step=100,
            xlb=xlabel, ylb=ylabel, title=name, ref_label='Exp. (Dickinson, J.M., P.E., 1967)')


def main(instruct):
    cwd = os.getcwd()
    pot_paths = glob.glob(os.path.join(cwd, 'zhuo*'))
    if instruct == 'lattice':
        lattice(pot_paths)
    elif instruct == 'elastic':
        elastic(pot_paths)


if __name__ == '__main__':
    intruct = sys.argv[1]
    main(intruct)
