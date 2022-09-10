from copy import deepcopy
import datetime
import glob
import os
import shutil
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import atomman as am
import atomman.unitconvert as uc
from multiprocessing import Pool
#print('atomman version =', am._version_)
#print('Notebook executed on', datetime.date.today())
flag = False

def plot_dd(data, dump, name):
    #base_system = am.load('atom_dump', 'disl_core.hcp.10-10.hennig.dump')
    base_system = am.load('atom_data', data)
    disl_system = am.load('atom_dump', dump)
    #base_system.pbc = (True, False, False)
    #disl_system.pbc = (True, False, False)
    print(base_system.natoms, disl_system.natoms)

    alat = uc.set_in_units(3.16236726751779, 'Å')
    burgers = np.array([alat*3**0.5/2, 0.0, 0.0])

    neighbors = base_system.neighborlist(cutoff = alat)
    #neighs = neighbors[1]
    #print(neighs)
    print(neighbors)
    #dd = am.defect.DifferentialDisplacement(base_system, disl_system)
    dd = am.defect.DifferentialDisplacement(base_system, disl_system, neighbors=neighbors, reference=0)
    #neighbors=neighbors, cutoff=0.9, reference=1)
    #print(dd)
    ddmax = np.linalg.norm(burgers) / 2

    params = {}
    params['plotxaxis'] = 'y'
    params['plotyaxis'] = 'z'
    params['xlim'] = (90, 110)
    params['ylim'] = (90, 110)
    #params['zlim'] = (-0.01, alat + 0.01)
    params['zlim'] = (-0.01, alat*3**0.5/2 + 0.01)
    params['figsize'] = 10
    params['arrowwidth'] = 1/50
    params['arrowscale'] = 1.2
    params['atomcmap'] = 'Greys'


    dd.plot(burgers, ddmax, **params)
    #dd.plot('x', ddmax)
    #plt.title('x component')
    fig_name = f'{name}.pdf'
    plt.savefig(fig_name)
    fig = os.path.join(os.getcwd(), fig_name)
    return fig
                

def apply_parallel(restart, pot_name,
                   temp, data, fig_path):
    os.chdir(restart)
    lgv = restart.split('/')[-1].split('.')[-1]
    dump = 'dump.easy.relax_final'
    log = glob.glob(os.path.join(os.getcwd(), 'in.*.log'))[0]
    pe_final = 'pe_final_state.mod'
    name = f'{pot_name}_{temp}K_{lgv}'
    print(f'working on {name}')
    fig = plot_dd(data, dump, name)
    shutil.copyfile(fig, os.path.join(fig_path, name + '.pdf'))


def main(pot_path, instruct):
    if instruct == 'ddplot':
        if os.path.exists('figures'):
            shutil.rmtree('figures')
        os.mkdir('figures')
        figs_path = os.path.join(os.getcwd(), 'figures')
    elif instruct == 'energy':
        if os.path.exists('energy'):
            shutil.rmtree('energy')
        os.mkdir('energy')
        en_path = os.path.join(os.getcwd(), 'energy')
    for pot in pot_path:
        pot_name = pot.split('/')[-1]
        os.chdir(pot)
        os.chdir('disl_core_easy_cylinder')
        path_temp = os.getcwd()
        for temp in ['300', '600']:
            os.chdir(path_temp)
            os.chdir(f'{temp}K_nose_hoover/Mo/bcc_bcc_110_screw_a')
            data = glob.glob(os.path.join(os.getcwd(), 'disl_core.*.init_sc.lmp'))[0]
            #print(data)
            fig_path_name = f'{pot_name}_{temp}K'
            if instruct == 'ddplot':
                fig_path = os.path.join(figs_path, fig_path_name)
                os.mkdir(fig_path)
            if instruct == 'energy':
                fig_path = None
            os.chdir('opti_structure')
            path_lgv = os.getcwd()
            restart_paths = glob.glob(os.path.join(path_lgv, 'restart.eq_lgv.*'))
            restart_paths.sort()
            if instruct == 'ddplot':
                processes = len(restart_paths)
                pool = Pool(processes=processes)
                print("Running via %d processes" % processes)
                res = []
                for restart in restart_paths:
                    res = pool.apply_async(apply_parallel,
                                           (restart, pot_name, temp,
                                            data, fig_path))
                pool.close()
                pool.join()
            elif instruct == 'energy':
                result = []
                for restart in restart_paths:
                    os.chdir(restart)
                    lgv = restart.split('/')[-1].split('.')[-1]
                    dump = 'dump.easy.relax_final'
                    log = glob.glob(os.path.join(os.getcwd(), 'in.*.log'))[0]
                    pe_final = 'pe_final_state.mod'
                    name = f'{pot_name}_{temp}K_{lgv}'
                    print(f'working on {name}')
                    with open(pe_final, 'r') as f1:
                        content = f1.readline()
                        pe = content.split(' ')[-1]
                    with open(log, 'r') as f2:
                        content = f2.read()
                        atom = re.findall('steps with (\d+) atoms', content)[0]
                    result.append(f'{lgv} {atom} {pe}')
                temp_name = f'{pot_name}_{temp}K'
                with open(os.path.join(en_path, temp_name), 'w') as f:
                    for ii in result:
                        f.write(ii)



if __name__ == '__main__':
    main_path = os.getcwd()
    pot_path = glob.glob(os.path.join(main_path, 'zhuo*'))
    print(pot_path)
    instruction = sys.argv[1]
    main(pot_path, instruction)
