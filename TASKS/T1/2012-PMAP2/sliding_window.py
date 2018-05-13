#
# APPLY SLIDING WINDOW
#

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import fabs
from os import listdir
from os.path import isfile, join

HEADER_TS  = 'ts'

HEADER_ACC_X = 'acc1_x'
HEADER_ACC_Y = 'acc1_y'
HEADER_ACC_Z = 'acc1_z'

HEADER_GYR_X = 'gyr_x'
HEADER_GYR_Y = 'gyr_y'
HEADER_GYR_Z = 'gyr_z'

HEADER_MAG_X = 'mag_x'
HEADER_MAG_Y = 'mag_y'
HEADER_MAG_Z = 'mag_z'

HEADER_USER = 'subject'
HEADER_LABEL = 'label'
HEADER_POSITION = 'position'

def apply_window_v2(array, window, uncertainty):
    old_ts = array.iloc[0][HEADER_TS];
    i = 0;
    values = []
    values_accX = []
    values_accY = []
    values_accZ = []

    values_gyrX = []
    values_gyrY = []
    values_gyrZ = []

    values_magX = []
    values_magY = []
    values_magZ = []
    out = pd.DataFrame()

    for index, line in array.iterrows():
        ts = line[HEADER_TS]       
        #print("ts:", ts)
        accX = line[HEADER_ACC_X]
        accY = line[HEADER_ACC_Y]
        accZ = line[HEADER_ACC_Z]

        magX = line[HEADER_GYR_X]
        magY = line[HEADER_GYR_Y]
        magZ = line[HEADER_GYR_Z]

        gyrX = line[HEADER_MAG_X]
        gyrY = line[HEADER_MAG_Y]
        gyrZ = line[HEADER_MAG_Z]

        diff = ts - old_ts;
        # add values  values += [ts]
        values += [ts]
        values_accX += [accX]
        values_accY += [accY]
        values_accZ += [accZ]

        values_gyrX += [gyrX]
        values_gyrY += [gyrY]
        values_gyrZ += [gyrZ]

        values_magX += [magX]
        values_magY += [magY]
        values_magZ += [magZ]
        #
        if diff == window or fabs(diff - window) <= uncertainty:
            i+=1
            old_ts = ts
            out = out.append(pd.DataFrame([{
         'ts': line[HEADER_TS],
          'accX': values_accX, 'accY': values_accY, 'accZ': values_accZ,
          'magX': values_magX, 'magY': values_magY, 'magZ': values_magZ,
          'gyrX': values_gyrX, 'gyrY': values_gyrY, 'gyrZ': values_gyrZ,
          'userID': line[HEADER_USER],
          'position': line[HEADER_POSITION], 
          'label': line[HEADER_LABEL]
          }]));

            ### set values  values = [old_ts]
            values = [old_ts]
            values_accX = [accX]
            values_accY = [accY]
            values_accZ = [accZ]
            values_gyrX = [gyrX]
            values_gyrY = [gyrY]
            values_gyrZ = [gyrZ]
            values_magX = [magX]
            values_magY = [magY]
            values_magZ = [magZ]
            continue
        elif diff > window or old_ts == -1:
            old_ts = ts
            # set values values = [old_ts]
            values = [old_ts]
            values_accX = [accX]
            values_accY = [accY]
            values_accZ = [accZ]
            values_gyrX = [gyrX]
            values_gyrY = [gyrY]
            values_gyrZ = [gyrZ]
            values_magX = [magX]
            values_magY = [magY]
            values_magZ = [magZ]
            
    return out;      

def run_sliding_window(window = 5, uncertainty=1):
    mypath = "./individual-data"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for filename in files:
        content = pd.read_csv(mypath + "/" + filename , sep=';')
        print(filename, len(content), content.columns)
        content_with_window = pd.DataFrame();
        users = set(content[HEADER_USER])
        for user in users:
                subject = content[content[HEADER_USER].isin([user])]
                print("len(subject):", len(subject))
                #experiments
                labels = set(content[HEADER_LABEL])
                print(labels)
                for label in labels:     
                    subject_exp = subject[subject[HEADER_LABEL].isin([label])]    
                    #positions
                    positions = list(set(subject_exp[HEADER_POSITION]))
                    for position in positions:
                        subject_exp_pos = subject_exp[subject_exp[HEADER_POSITION].isin([position])]
                        subject_pos_label = subject_exp_pos.sort_values([HEADER_TS])
                        print("position:", position, " label:", label)
                        old_ts = subject_pos_label.iloc[0]['ts']
                        out = apply_window_v2(subject_pos_label, window, uncertainty)
                        print(len(out))
                        content_with_window = content_with_window.append(out)

        content_with_window.to_csv('./sliding_window_data/' + str(window) + "s_" + filename, sep=';')
        print("len(content):", len(content), "len(content_with_window):", len(content_with_window))
        print("len(content.columns):", len(content.columns), "len(content_with_window.columns):", len(content_with_window.columns))
    
run_sliding_window(3, 0.5)