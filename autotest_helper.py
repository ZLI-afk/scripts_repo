import os
import sys
import shutil
import pickle

strategy_list_str = [
    '01_2_g100_v10_c10',
    '01_2_g20_v10_c10',
    '01_2_g50_v10_c10',
    '01_3_g100_v10_c10',
    '01_3_g20_v10_c10',
    '01_3_g50_v10_c10'
]


def save_v(v, filename):
    if os.path.isfile(filename):
        print(f'{filename} already exist, removing it...')
        os.remove(filename)
    with open(filename, 'wb') as f:
        pickle.dump(v, f)


def load_v(filename):
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r


def check_input():
    if not all([os.path.isfile('param_relax.json'),
                os.path.isfile('param_prop.json'),
                os.path.isfile('POSCAR.bcc'),
                os.path.isfile('POSCAR.bcc')]):
        print('!!!missing necessary input files\n'
              'do check if all followed file exist:\n'
              'param_relax.json\n'
              'param_prop.json\n'
              'POSCAR.bcc\n'
              'POSCAR.fcc\n'
              'will exit now!')
        exit()
    else:
        return


def return_all_strategy():
    strategy_list = []
    for ii in os.listdir():
        if os.path.isdir(ii) and os.path.islink(ii):
            strategy_list.append(ii)
    return strategy_list


def dump_job_relax(job_name):
    job_relax = [
        '#!/usr/bin/env bash\n',
        '\n',
        f'#SBATCH --job-name="{job_name}_relax"\n',
        '#SBATCH --partition=gpu\n',
        '#SBATCH --qos=gpu\n',
        '#SBATCH --ntasks=4\n',
        '#SBATCH --nodes=1\n',
        '#SBATCH --gres=gpu:1\n',
        '#SBATCH --time=1-00:00:00\n',
        '#SBATCH --error="j%j.stderr"\n',
        '#SBATCH --output="j%j.stdout"\n',
        'hostname > ./hostname\n',
        'nvidia-smi > ./nvidia-smi\n',
        '#module load dp/gpu/2.0.1\n',
        'for kk in 00?\n',
        'do\n',
        '    cd $kk/confs\n',
        '    for ii in std-*\n',
        '    do\n',
        '        echo $ii\n',
        '        cd $ii/relaxation/relax_task\n',
        '        lmp -i in.lammps -v restart 0\n',
        '        cd ../../..\n',
        '    done\n',
        '    cd ../..\n',
        'done'
    ]
    with open('job_relax', 'w') as f:
        for ii in range(len(job_relax)):
            f.write(job_relax[ii])


def dump_job_prop(job_name, structs, props):
    job_prop = [
        '#!/usr/bin/env bash\n',
        '\n',
        f'#SBATCH --job-name="{job_name}"\n',
        '#SBATCH --partition=gpu\n',
        '#SBATCH --qos=gpu\n',
        '#SBATCH --ntasks=4\n',
        '#SBATCH --nodes=1\n',
        '#SBATCH --gres=gpu:1\n',
        '#SBATCH --time=3-00:00:00\n',
        '#SBATCH --error="j%j.stderr"\n',
        '#SBATCH --output="j%j.stdout"\n',
        'hostname > ./hostname\n',
        'nvidia-smi > ./nvidia-smi\n',
        'for tt in 00?\n',
        'do\n',
        '    cd $tt/confs\n',
        f'    for kk in {structs}\n',
        '    do\n',
        '        cd $kk\n',
        f'        for jj in {props}\n',
        '        do\n',
        '            cd $jj\n',
        '            for ii in task.*\n',
        '            do\n',
        '                cd $ii\n',
        '                lmp -i in.lammps -v restart 0\n',
        '                cd ..\n',
        '            done\n',
        '            cd ..\n',
        '        done\n',
        '        cd ..\n',
        '    done\n',
        '    cd ../..\n',
        'done'
    ]
    with open('job_prop', 'w') as f:
        for ii in range(len(job_prop)):
            f.write(job_prop[ii])


def make_init_dirs(strategy_list,
                   param_relax, param_prop,
                   poscar_bcc, poscar_fcc):
    cwd = os.getcwd()
    os.mkdir('autotests')
    main_path = os.path.join(cwd, 'autotests')
    main_path_abs = os.path.abspath(main_path)
    os.chdir(main_path_abs)

    orig_strategy_list_abs = []
    strategy_list_abs = []
    model_list_abs = []
    for ii in strategy_list:
        os.mkdir(ii)
        orig_strategy_abs = os.path.abspath(f'../{ii}')
        orig_strategy_list_abs.append(orig_strategy_abs)
        os.chdir(ii)
        cur_strategy = os.path.abspath(os.getcwd())
        strategy_list_abs.append(cur_strategy)
        dump_job_relax(ii)

        for jj in ['000', '001', '002', '003']:
            os.mkdir(jj)
            os.chdir(jj)
            cur_model_path = os.path.abspath(os.getcwd())
            model_list_abs.append(cur_model_path)
            print(f'working on {ii}/{jj}...')
            os.symlink(param_relax, './param_relax.json')
            os.symlink(param_prop, './param_prop.json')
            orig_model_path = os.path.join(orig_strategy_abs, jj)
            os.symlink(os.path.join(orig_model_path, 'frozen_model.pb'), './frozen_model.pb')
            os.mkdir('confs')
            os.chdir('confs')
            confs = os.getcwd()
            os.makedirs('std-bcc')
            os.chdir('std-bcc')
            os.symlink(poscar_bcc, './POSCAR')
            os.chdir(confs)
            os.mkdir('std-fcc')
            os.chdir('std-fcc')
            os.symlink(poscar_fcc, './POSCAR')
            os.chdir(cur_strategy)

        os.chdir(main_path_abs)
    return strategy_list_abs, model_list_abs


def make_relax(model_list):
    for ii in model_list:
        os.chdir(ii)
        print(f'working on direction: {ii}')
        os.system('dpgen autotest make param_relax.json')


def make_prop(model_list):
    for ii in model_list:
        os.chdir(ii)
        print(f'working on direction: {ii}')
        os.system('dpgen autotest make param_prop.json')


def run_relax(strategy_list):
    for ii in strategy_list:
        os.chdir(ii)
        print(f'working on direction: {ii}')
        os.system('sbatch job_relax')


def run_prop(strategy_list, structs, props):
    for ii in strategy_list:
        os.chdir(ii)
        job_name = ii.split('/')[-1]
        print(f'working on direction: {ii}')
        dump_job_prop(job_name, structs, props)
        os.system('sbatch job_prop')


def post_relax(model_list):
    for ii in model_list:
        os.chdir(ii)
        print(f'working on direction: {ii}')
        os.system('dpgen autotest post param_relax.json')


def post_prop(model_list):
    for ii in model_list:
        os.chdir(ii)
        print(f'working on direction: {ii}')
        os.system('dpgen autotest post param_prop.json')


def main(param_relax, param_prop,
         poscar_bcc, poscar_fcc,
         strategy_list_path, model_list_path):
    if sys.argv[1] == 'make_file':
        print('->> start make_file step <<-')
        if os.path.exists('autotests'):
            print('autotests direction already exist...\n')
            is_rm = input('remove it?(y/n): ')
            if is_rm == 'y':
                shutil.rmtree('autotests')
                print('removed...')
            else:
                print('will exit')
                exit()
        get_strategy = input('run all potential listed in current direction? (y/n): ')
        if sys.argv[2] == 'y':
            strategy_list_str = return_all_strategy()
        strategy_list, model_list = make_init_dirs(strategy_list_str,
                                                   param_relax, param_prop,
                                                   poscar_bcc, poscar_fcc)
        save_v(strategy_list, 'strategy_list')
        save_v(model_list, 'model_list')
        print('-<< finished! >>-')

    elif sys.argv[1] == 'make_relax':
        print('->> start make_relax step <<-')
        try:
            model_list = load_v(model_list_path)
        except:
            print('run make_relax first!\nwill exit!')
            exit()
        else:
            make_relax(model_list)
            print('-<< finished! >>-')

    elif sys.argv[1] == 'run_relax':
        print('->> start run_relax step <<-')
        try:
            strategy_list = load_v(strategy_list_path)
        except:
            print('run make_relax first!\nwill exit!')
            exit()
        else:
            run_relax(strategy_list)
            print('-<< finished! >>-')

    elif sys.argv[1] == 'post_relax':
        print('->> start post_relax step <<-')
        try:
            model_list = load_v(model_list_path)
        except:
            print('run make_relax first!\nwill exit!')
            exit()
        else:
            post_relax(model_list)
            print('-<< finished! >>-')

    elif sys.argv[1] == 'make_prop':
        print('->> start make_prop step <<-')
        try:
            model_list = load_v(model_list_path)
        except:
            print('run make_prop first!\nwill exit!')
            exit()
        else:
            make_prop(model_list)
            print('-<< finished! >>-')

    elif sys.argv[1] == 'run_prop':
        print('->> start run_prop step <<-')
        try:
            strategy_list = load_v(strategy_list_path)
        except:
            print('run make_prop first!\nwill exit!')
            exit()
        else:
            get_prop = input('please indicate property list to run: ')
            if get_prop == 'all':
                props = '{elastic_00,eos_00,surface_00,vacancy_00,interstitial_00,gamma_00}'
            else:
                props = get_prop
            get_struct = input('please indicate configure type to run: ')
            run_prop(strategy_list,
                     structs=get_struct,
                     props=props)
            print('-<< finished! >>-')

    elif sys.argv[1] == 'post_prop':
        print('->> start post_prop step <<-')
        try:
            model_list = load_v(model_list_path)
        except:
            print('run make_relax first!\nwill exit!')
            exit()
        else:
            post_prop(model_list)
            print('-<< finished! >>-')

    else:
        print('!!wrong input argv!\n' +
              '!!only support:\n' +
              '     make_file\n' +
              '     make_relax\n' +
              '     run_relax\n' +
              '     post_relax\n' +
              '     make_prop\n' +
              '     run_prop\n' +
              '     post_prop\n' +
              '!!please input in order')


if __name__ == "__main__":
    cwd = os.getcwd()
    check_input()
    param_relax = os.path.join(cwd, 'param_relax.json')
    param_prop = os.path.join(cwd, 'param_prop.json')
    poscar_bcc = os.path.join(cwd, 'POSCAR.bcc')
    poscar_fcc = os.path.join(cwd, 'POSCAR.fcc')
    main_path = os.path.join(cwd, 'autotests')
    strategy_list_path = os.path.join(main_path, 'strategy_list')
    model_list_path = os.path.join(main_path, 'model_list')

    main(param_relax, param_prop,
         poscar_bcc, poscar_fcc,
         strategy_list_path, model_list_path)
