import random

import numpy as np
import pandas as pd

import psycopg as psql

import configparser

config = configparser.ConfigParser()

# %%


ADC_channels = ['P1_1', 'P1_2', 'P2_1', 'P2_2', 'P3_1', 'P3_2', 'P4_1', 'P4_2', 'P5_1', 'P5_2']
IMU_channels = ['Euler_x', 'Euler_y', 'Euler_z', 'Acc_x', 'Acc_y', 'Acc_z']

sign_types = ['static', 'dynamic']
sign_types_dict = {'a': sign_types[0],
                   'ą': sign_types[1],
                   'b': sign_types[0],
                   'c': sign_types[0],
                   'ć': sign_types[1],
                   'ch': sign_types[1],
                   'cz': sign_types[1],
                   'd': sign_types[1],
                   'e': sign_types[0],
                   'ę': sign_types[1],
                   'f': sign_types[1],
                   'g': sign_types[1],
                   'h': sign_types[1],
                   'i': sign_types[0],
                   'j': sign_types[1],
                   'k': sign_types[1],
                   'l': sign_types[0],
                   'ł': sign_types[1],
                   'm': sign_types[0],
                   'n': sign_types[0],
                   'ń': sign_types[1],
                   'o': sign_types[0],
                   'ó': sign_types[1],
                   'p': sign_types[0],
                   'r': sign_types[0],
                   'rz': sign_types[1],
                   's': sign_types[0],
                   'ś': sign_types[1],
                   'sz': sign_types[1],
                   't': sign_types[0],
                   'u': sign_types[0],
                   'w': sign_types[0],
                   'y': sign_types[0],
                   'z': sign_types[1],
                   'ź': sign_types[1],
                   'ż': sign_types[1]}

config.read('config.ini')
# %%

with psql.connect(dbname=config['DB']['dbname'], user=config['DB']['user'], password=config['DB']['password'], host=config['DB']['dbpi_ip_addr']) as psqlconn:
    psqlcur = psqlconn.cursor()

    psqlcur.execute('SELECT * FROM static_gestures;')
    stat_gest = pd.DataFrame(psqlcur.fetchall())
    psqlcur.execute('SELECT * FROM dynamic_gestures;')
    dyn_gest = pd.DataFrame(psqlcur.fetchall())
    psqlcur.execute('SELECT * FROM augmented_gestures;')
    aug_gest = pd.DataFrame(psqlcur.fetchall())

gest_base_buf=pd.concat([stat_gest.iloc[:,1:],dyn_gest.iloc[:,1:]],ignore_index=True)
gest_base_buf.columns=range(gest_base_buf.shape[1])
gest_base=pd.concat([gest_base_buf,aug_gest],ignore_index=True)
gest_base.columns=(*ADC_channels,*IMU_channels,'sign','timedelta','id')




