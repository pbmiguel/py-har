#
# APPLY SLIDING WINDOW
#

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import fabs
          
def apply_window_v2(array):
    old_ts = array.iloc[0]['Fts(ms)'];
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
        ts = line['Fts(ms)']       
        accX = line['accX']
        accY = line['accY']
        accZ = line['accZ']

        magX = line['magX']
        magY = line['magY']
        magZ = line['magZ']

        gyrX = line['gyrX']
        gyrY = line['gyrY']
        gyrZ = line['gyrZ']

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
        if diff == WINDOW or fabs(diff - WINDOW) <= UNCERTAINTY:
            i+=1
            old_ts = ts
            out = out.append(pd.DataFrame([{
         'Its(ms)': line['Its(ms)'], 'Fts(ms)': line['Fts(ms)'],
          'accX': values_accX, 'accY': values_accY, 'accZ': values_accZ,
          'magX': values_magX, 'magY': values_magY, 'magZ': values_magZ,
          'gyrX': values_gyrX, 'gyrY': values_gyrY, 'gyrZ': values_gyrZ,
          'userGender': line['userGender'], 'userAge': line['userAge'], 'userID': line['userID'],
          'position': line['position'], 
          'label': line['label'],
          'filename': line['filename'],
          'experiment': line['experiment']}]));

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
        elif diff > WINDOW or old_ts == -1:
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
'''
all_values = pd.read_csv("./t1-3_all_values_50ms.csv");
#subject = all_values[all_values['userID'].isin(['Subject 01'])]
#subject_pos = subject[subject['position'].isin(['ankle'])]
#subject_pos_label = subject_pos[subject_pos['experiment'].isin(['Walking_3'])]
#subject_pos_label = subject_pos_label.sort_values(['Fts(ms)'])
print("len(all_values):", len(all_values))
print("len(all_values.columns):", len(all_values.columns))
WINDOW = 2000;
UNCERTAINTY = 500;
users = list(set(all_values['userID']));
all_values_with_window = pd.DataFrame();
for line in users:
        print(line)
        subject = all_values[all_values['userID'].isin([line])]
        print("len(subject):", len(subject))
        #experiments
        experiments = list(set(subject['experiment']))
        for lineExp in experiments:     
            subject_exp = subject[subject['experiment'].isin([lineExp])]    
            #positions
            positions = list(set(subject_exp['position']))
            for linePos in positions:
                subject_exp_pos = subject_exp[subject_exp['position'].isin([linePos])]
                subject_pos_label = subject_exp_pos.sort_values(['Fts(ms)'])
                # applying sliding window
                out = apply_window_v2(subject_pos_label);
                all_values_with_window = all_values_with_window.append(out);

#all_values_with_window.to_csv("t1-42_all_values_with_window_" + str(WINDOW) + "_ms.csv");
print("len(all_values_with_window):", len(all_values_with_window))
print("len(all_values_with_window.columns):", len(all_values_with_window.columns))
#print(out.head(5)['magY'])
#print()
#print(subject_pos_label.head(5)['magY'])
'''

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
                print("type(window_values):", type(window_values))
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