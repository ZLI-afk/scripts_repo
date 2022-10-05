import os
import sys
import glob
import json
import shutil
from monty.serialization import loadfn, dumpfn

ntasks = 4
seeds = [10 ** (ii + 1) for ii in range(ntasks)]
train_sets = '/home/zhuoyli/labspace/long_train/Mo_dpdata'

def dump_job(job_name) -> None:
    job = [
        '#!/usr/bin/env bash\n',
        '\n',
        f'#SBATCH --job-name="{job_name}_train"\n',
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
        '#module load dp/gpu/2.0.1\n',
        f'dp train input_{job_name}.json 1>train.log 2>train.err &&\n',
        'dp freeze -o frozen_model.pb'
    ]
    with open('job', 'w') as f:
        for ii in range(len(job)):
            f.write(job[ii])


def make_dirs(strategies):
    for strategy in strategies:
        print(f'working on {strategy}')
        os.chdir(strategy)
        cwd = os.getcwd()
        train_set_name = train_sets.split('/')[-1]
        if os.path.islink(train_set_name):
            os.remove(train_set_name)
        os.symlink(train_sets, train_set_name)
        input_json = glob.glob(os.path.join(cwd, 'input*.json'))
        input_param = loadfn(input_json[0])
        for i in range(ntasks):
            os.chdir(cwd)
            model_name = f'00{i}'
            if os.path.exists(model_name):
                shutil.rmtree(model_name)
            os.mkdir(model_name)
            print(model_name)
            model = os.path.join(cwd, model_name)
            os.chdir(model)
            if input_param["model"]["descriptor"]["type"] == "se_e2_a":
                input_param["model"]["descriptor"]["seed"] = seeds[i]
                input_param["model"]["fitting_net"]["seed"] = seeds[i]
                input_param["training"]["seed"] = seeds[i]
            elif input_param["model"]["descriptor"]["type"] == "hybrid":
                input_param["model"]["descriptor"]["list"][0]["seed"] = seeds[i]
                input_param["model"]["descriptor"]["list"][1]["seed"] = seeds[i]
            dump_job(model_name)
            with open(f'input_{model_name}.json', 'w') as f:
                json.dump(input_param, f, indent=4)


def run(strategies):
    for ii in strategies:
        os.chdir(ii)
        cwd = os.getcwd()
        models = glob.glob(os.path.join(cwd, '00?'))
        for jj in models:
            os.chdir(jj)
            os.system('sbatch job')


def init(strategies):
    for ii in strategies:
        os.chdir(ii)
        cwd = os.getcwd()
        models = glob.glob(os.path.join(cwd, '00?'))
        for jj in models:
            os.chdir(jj)
            module_name = jj.split('/')[-1]
            input_name = f'input_{module_name}.json'
            with open('job', 'r') as f:
                context = f.readlines()
                context[14] = f'dp train --init-model model.ckpt ' \
                              f'{input_name} 1>train.log 2>train.err &&\n'
            with open('job', 'w') as f1:
                for line in context:
                    f1.write(line)
            input_param = loadfn(input_name)
            input_param['learning_rate']['start_lr'] = 0.0001
            with open(input_name, 'w') as f2:
                json.dump(input_param, f2, indent=4)

def main(strategies):
    if sys.argv[1] == 'make':
        make_dirs(strategies)
    elif sys.argv[1] == 'run':
        run(strategies)
    elif sys.argv[1] == 'init':
        init(strategies)


if __name__ == "__main__":
    cwd = os.getcwd()
    cur_files = glob.glob(os.path.join(cwd, '*'))
    strategies = [ii for ii in cur_files if os.path.isdir(ii)]
    main(strategies)
    print('finished')
