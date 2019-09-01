import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import fabs
from utils import ske, skew, kur, avg, maxi, mini, aco, cec, cen, cor, his, iqr, iqrr, kur, kurtosis, mad, median_absolute_deviation, var, std, rms
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

def extract_features(data, fefunctions):
    out = data.copy();
    # data = [ [1,2] , [2,3]]
    # columns = ['accX', 'accY', ['accX', 'accY']]
    # fefunction = [avg, ['accX', 'accY', 'accZ']]
    for i in fefunctions:
        columns = i[1]
        fefunction = i[0]
        for selected_columns in columns:
            #name = "avg_";
            name = str(fefunction.__name__)
            if isinstance(selected_columns, list):
                array = pd.DataFrame();
                for selected_column in selected_columns:
                    name += "_" + str(selected_column);
                    array = array.assign(**{selected_column: data[selected_column].values})
            else:
                try:
                    array = pd.DataFrame(data[selected_columns])
                    name += "_" + str(selected_columns);
                except:
                    print("Invalid Column", str(selected_columns) , " !!!");
                    continue;
            #
            new_column = pd.DataFrame(columns=[name]);
            # calculate         
            for index, line in array.iterrows():
                window_values = []
                nr_columns = len(array.columns)
                for column in range(0, nr_columns):
                    window_values += [line[column]]
                # mean
                #print("type(window_values):", type(window_values))
                mean = fefunction(window_values)
                #print(mean)
                new_column = new_column.append(pd.DataFrame({name: [mean]}), ignore_index=True);
            #
            #print("len(new_column):", len(new_column))
            out = out.assign(**{name: new_column[name].values})
    return out;

'''
### 
all_values = all_values_with_window;
all_values_with_fe = pd.DataFrame()
users = list(set(all_values['userID']));
#users.sort()
# users
for line in users:
    subject = all_values[all_values['userID'].isin([line])]
    #experiments
    experiments = list(set(subject['experiment']))
    print(line)
    for lineExp in experiments:     
        subject_exp = subject[subject['experiment'].isin([lineExp])]    
        #positions
        positions = list(set(subject_exp['position']))
        for linePos in positions:
            subject_exp_pos = subject_exp[subject_exp['position'].isin([linePos])]
            subject_pos_label = subject_exp_pos.sort_values(['Fts(ms)'])
            # applying feature extraction
            print("subject_pos_label:", type(subject_pos_label))
            all_values_with_fe = all_values_with_fe.append(extract_features(subject_pos_label, [ 
                                # statistical 
                                [ske, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ', 
                                       'magX', 'magY', 'magZ']],
    
                                [kur, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ', 
                                       'magX', 'magY', 'magZ']],
                                # time series
                                [avg, ['accX', 'accY', 'accZ', ['accX', 'accY', 'accZ'],
                                       'gyrX', 'gyrY', 'gyrZ', ['gyrX', 'gyrY', 'gyrZ'],
                                       'magX', 'magY', 'magZ', ['magX', 'magY', 'magZ']]],
    
                                [maxi, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [mini, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [var, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [cen, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],  
    
                                [std, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [rms, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [iqr, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],
    
                                [mad, ['accX', 'accY', 'accZ',
                                       'gyrX', 'gyrY', 'gyrZ',
                                       'magX', 'magY', 'magZ']],    
    
                                [cor, [ ['accX', 'accY'], ['accY', 'accZ'], ['accX', 'accZ'],
                                      ['gyrX', 'gyrY'], ['gyrY', 'gyrZ'], ['gyrX', 'gyrZ'],
                                      ['magX', 'magY'], ['magY', 'magZ'], ['magX', 'magZ']]],
    
                                #[aco, ['accX', 'accY', 'accZ',
                                #       'gyrX', 'gyrY', 'gyrZ',
                                #       'magX', 'magY', 'magZ']]    
                              ]));
            
            
all_values_with_fe.to_csv("all_values_with_fe_" + str(WINDOW) + "_ms.csv");
'''

def run_feature_extraction(window=5, uncertainty=1):
    mypath = "./individual-data/"
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    i = 0
    for filename in files:
        print(filename)
        content = pd.read_csv(mypath + filename, sep=';')
        print(filename, content.dtypes, content.columns)
        out = pd.DataFrame();
        users = set(content[HEADER_USER])
        for user in users:
                subject = content[content[HEADER_USER].isin([user])]
                #print("len(subject):", len(subject))
                #experiments
                labels = set(content[HEADER_LABEL])
                print(labels)
                for label in labels:     
                    subject_exp = subject[subject[HEADER_LABEL].isin([label])]    
                    #positions
                    positions = list(set(subject_exp[HEADER_POSITION]))
                    for position in positions:
                        try:
                            subject_exp_pos = subject_exp[subject_exp[HEADER_POSITION].isin([position])]
                            subject_pos_label = subject_exp_pos.sort_values([HEADER_TS])
                            #print("position:", position, " label:", label)
                            subject_pos_label_window = apply_window_v2(subject_pos_label, window, uncertainty)
                            # applying feature extraction
                            #print("subject_pos_label:", type(subject_pos_label))
                            out = out.append(extract_features(subject_pos_label_window, [ 
                                                # statistical 
                                                [ske, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ', 
                                                    'magX', 'magY', 'magZ']],
                    
                                                [kur, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ', 
                                                    'magX', 'magY', 'magZ']],
                                                # time series
                                                [avg, ['accX', 'accY', 'accZ', ['accX', 'accY', 'accZ'],
                                                    'gyrX', 'gyrY', 'gyrZ', ['gyrX', 'gyrY', 'gyrZ'],
                                                    'magX', 'magY', 'magZ', ['magX', 'magY', 'magZ']]],
                    
                                                [maxi, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [mini, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [var, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [cen, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],  
                    
                                                [std, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [rms, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [iqr, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],
                    
                                                [mad, ['accX', 'accY', 'accZ',
                                                    'gyrX', 'gyrY', 'gyrZ',
                                                    'magX', 'magY', 'magZ']],    
                    
                                                [cor, [ ['accX', 'accY'], ['accY', 'accZ'], ['accX', 'accZ'],
                                                    ['gyrX', 'gyrY'], ['gyrY', 'gyrZ'], ['gyrX', 'gyrZ'],
                                                    ['magX', 'magY'], ['magY', 'magZ'], ['magX', 'magZ']]],
                    
                                                #[aco, ['accX', 'accY', 'accZ',
                                                #       'gyrX', 'gyrY', 'gyrZ',
                                                #       'magX', 'magY', 'magZ']]    
                                            ]));
                        except: 
                            continue;

        # drop NA values
        out = out.dropna(axis=0, how='any')
        out.to_csv('./feature_extraction_data/' + str(window) + "s_" + filename, sep=';')
        #print("len(content):", len(content), "len(content_with_window):", len(out))
        #print("len(content.columns):", len(content.columns), "len(content_with_window.columns):", len(out.columns))

run_feature_extraction(3, 0.5)