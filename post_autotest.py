from __future__ import print_function
from genericpath import isfile
import os
import sys
import shutil
import glob
import pickle
import numpy as np
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

out_path = os.path.join(os.getcwd(), 'figures')
eos_benchmark = '/home/zhuoyli/benchmarks/Mo/eos/bcc_dft'
# cohesive_benchmark = '/home/zhuoyli/benchmarks/Mo/eos/cohesive_spin'
# gamma_benchmark = '/home/zhuoyli/benchmarks/Mo/gamma/110_14l'
# gammaA_benchmark = '/home/zhuoyli/benchmarks/Mo/gamma/112_20l'
# gammaB_benchmark = '/home/zhuoyli/benchmarks/Mo/gamma/123_20l'

xlabel = ''
ylabel = ''



class PlotFig:
  def __init__(self):
    #self.output_pic_fname = name
    self.fontsize = 13
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

    self.markers = ['o','H','v','d','s','*','P','^']
    self.markers_size = [5,5,5,5,5,5,5,5,5,5]

  def plot(self, pathes: dict, pic_name='my_plot',
           step=100, xlb=xlabel, ylb=ylabel, title=None):
    ax = self.ax
    ii = 0
    for key in pathes:
        line = np.loadtxt(pathes[key], dtype=float)
        x = line[:, 0]
        y = line[:, 1]
        x_m = np.linspace(line[0, 0], line[-1, 0], step)
        Spline = interpolate.make_interp_spline(x, y)
        y_m = Spline(x_m)
        if key=='DFT':
            ax.plot(x, y, color='black', linewidth=2.8, zorder=100,
                    label=key, alpha=0.8, marker='s',
                    ms=6, mec='black', mfc='white')
        elif key=='Expt':
            ax.plot(x, y, color='black', zorder=100,
                    label=key, alpha=0.8, marker='d',
                    ms=6, mec='black', mfc='white')
        elif key=='DP':
            ax.plot(x, y, color='b', linewidth=2.8, zorder=99,
                    label=key, alpha=0.8, marker='o',
                    ms=6, mec='b', mfc='white')
        else:
            ii += 1
            #ax.plot(x_m, y_m, color=self.colors[ii], linewidth=2, zorder=ii,
            #        label=key, alpha=0.8, marker=self.markers[ii],
            #        ms=self.markers_size[ii], mec=self.colors[ii], mfc='white')
            ax.plot(x_m, y_m, color=self.colors[ii], linewidth=2, linestyle='-.', zorder=ii,
                    label=key, alpha=0.8, mec=self.colors[ii], mfc='white')

    ax.legend()
    if title:
        ax.set_title(title, fontsize=self.fontsize, fontweight='bold')

    if title in ['gamma_line_110', 'gamma_line_112', 'gamma_line_123']:
        xlb = 'Fault displacement along 1/2[111]'
        ylb = 'Fault energy ${E}$ (J/m$^{2}$)'
    elif title == 'cohesive_energy':
        xlb = 'Scaled lattice parameter a/a0'
        ylb = 'Cohesive energy Ec (eV/atom)'
        ax.set_xlim(0.6,2.2)
        ax.set_ylim(-13,10)
    elif title == 'lattice_finite_T':
        xlb = 'Temperature (K)'
        ylb = 'Lattice parameter (\AA)'
        ax.set_ylim(3.16,3.26)
    elif title == 'elastic_finite_T_c11':
        xlb = 'Temperature (K)'
        ylb = 'Elastic constant C11 (GPa)'
    elif title == 'elastic_finite_T_c12':
        xlb = 'Temperature (K)'
        ylb = 'Elastic constant C12 (GPa)'
    elif title == 'elastic_finite_T_c44':
        xlb = 'Temperature (K)'
        ylb = 'Elastic constant C44 (GPa)'
    elif title == 'peierls_barrier':
        xlb = 'Reaction coordinate'
        ylb = 'Energy barrier (meV/b)'
    ax.set_xlabel(xlb, fontsize=self.fontsize,fontweight='bold')
    ax.set_ylabel(ylb, fontsize=self.fontsize,fontweight='bold')
    ax.grid(True)
    plt.tight_layout()
    os.chdir(out_path)
    #plt.savefig(f'{pic_name}', transparent=True, bbox_inches='tight')
    plt.savefig(f'{pic_name}.pdf', transparent=True, bbox_inches='tight', dpi=300)


def main(files):
    cwd = os.getcwd()
    path_dict = {}
    if os.path.join(cwd, 'DFT') in files:
        dft = os.path.join(cwd, 'DFT')
        files.remove(dft)
        path_dict['DFT'] = dft
    if os.path.join(cwd, 'Expt') in files:
        expt = os.path.join(cwd, 'Expt')
        files.remove(expt)
        path_dict['Expt'] = expt
    if os.path.join(cwd, 'DP') in files:
        dp = os.path.join(cwd, 'DP')
        files.remove(dp)
        path_dict['DP'] = dp
    for file in files:
        name = file.split('/')[-1]
        path_dict[name] = file
    path_dict = dict(sorted(path_dict.items(), key=lambda x: x[0]))

    # plot
    pf = PlotFig()   
    pf.plot(path_dict, pic_name=sys.argv[1], title=sys.argv[1]) 

if __name__ == '__main__':
    dir_name = sys.argv[1]
    main_path = os.getcwd()
    if not os.path.exists(dir_name):
        print(f'{dir_name} not exits!, will exit!')
        exit()
    else:
        dir = os.path.join(main_path, dir_name)
        os.chdir(dir)
        files = glob.glob(os.path.join(dir, '*'))
        main(files)