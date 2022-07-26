import os
import sys
import shutil
import pickle
import numpy as np
import pandas as pd
import matplotlib as plt

def save_v(v,filename):
    if os.path.isfile(filename):
        print(f'{filename} already exist, removing it...')
        os.remove(filename)
    with open(filename, 'wb') as f:
        pickle.dump(v, f)

def load_v(filename):
    with open(filename, 'rb') as f:
        r = pickle.load(f)
    return r

def elastic():
    pass

def vacancy():
    pass

def interstitial():
    pass

def surface():
    pass

def eos():
    pass

def gamma():
    pass

def main(props, strategy_list, model_list):
    for prop in props:
        if prop == 'elastic':
            elastic()
        elif prop == 'vacancy':
            vacancy()
        elif prop == 'interstitial':
            interstitial()
        elif prop == 'surface':
            surface()
        elif prop == 'eos':
            eos()
        elif prop == 'gamma':
            gamma()
        else:
            print(f'error! input unsupported property type: {prop}')
            print('will exit...')

if __name__ == '__main__':
    cwd = os.getcwd()
    if not os.path.exists('autotests'):
        print('post_results direction already exist...\n')
        is_rm = input('remove it?(y/n): ')
        if is_rm == 'y':
            shutil.rmtree('post_results')
            print('removed...')
        else:
            print('will exit')
            exit()
    if sys.argv[1] == 'all':
        props = [elastic, eos, surface, vacancy, interstitial, gamma]
    else:
        props = sys.argv[1]


    try:
        strategy_list = load_v(strategy_list)
        model_list = load_v(model_list)
    except:
        print('missing path dump files: strategy_list and model_list\n' +
              'please follow steps of autotest_helper.py first\n'+
              'will exit...')
        exit()
    else:
        main(props)


