'''

header
    Ts(s);Label; BPM;4-20 IMU Hand;21-37 IMU Chest;38-54 IMU Ankle;

#- The IMU sensory data contains the following columns:  
#– 1 temperature (°C)  
#– 2-4 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit  
#– 5-7 3D-acceleration data (ms-2), scale: ±6g, resolution: 13-bit  
#– 8-10 3D-gyroscope data (rad/s)  
#– 11-13 3D-magnetometer data (μT)  
#– 14-17 orientation (invalid in this data collection) 

Labels
    – 1 lying
    – 2 sitting
    – 3 standing
    – 4 walking
    – 5 running
    – 6 cycling
    – 7 Nordic walking
    – 9 watching TV
    – 10 computer work
    – 11 car driving
    – 12 ascending stairs
    – 13 descending stairs
    – 16 vacuum cleaning
    – 17 ironing
    – 18 folding laundry
    – 19 house cleaning
    – 20 playing soccer
    – 24 rope jumping
    – 0 other (transient activities)

Labels Per User:
    subject101.dat
        {'running', 'rope jumping', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject102.dat
        {'running', 'rope jumping', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject103.dat
        {'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'ascending stairs', 'lying', 'other'}
    subject104.dat
        {'running', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject105.dat
        {'running', 'rope jumping', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject106.dat
        {'running', 'rope jumping', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject107.dat
        {'running', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject108.dat
        {'running', 'rope jumping', 'ironing', 'sitting', 'standing', 'walking', 'vacuum clearning', 'descending stairs', 'nordic walking', 'ascending stairs', 'cycling', 'lying', 'other'}
    subject109.dat
        {'rope jumping', 'other'}
'''
import csv

# read flash.dat to a list of lists
#datContent = [i.strip().split() for i in open("C:\\Users\\paulo\\Documents\\py-har\\datasets\\raw-files\\PAMAP2-2012\\PAMAP2_Dataset\\Protocol\\subject101.dat").readlines()]

## write it as a new CSV file
#with open("./pmap2-subject109.csv", "wb") as f:
##    writer = csv.writer(f)
 #   writer.writerows(datContent)
def get_col(arr, col):
    return list(map(lambda x : x[col], arr))

#print(len(datContent[0]))
#print(datContent[1][1])
#print(set(get_col(datContent, 1)))

def read_files_from_directory():
    mypath = "C:\\Users\\paulo\\Documents\\py-har\\datasets\\raw-files\\PAMAP2-2012\\PAMAP2_Dataset\\Protocol"
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for filename in files:
        print(filename)
        #print(datContent[1][1])
        datContent = [i.strip().split() for i in open(mypath+ "\\" + filename).readlines()]
        print(set(get_col(datContent, 1)))

HEADER_TS = 0
HEADER_LABEL = 1
HEADER_BPM = 2
HEADER_IMU_HAND = 3
HEADER_IMU_CHEST = 20
HEADER_IMU_ANKLE = 37
HEADER_TEMP = 0
HEADER_ACC_1 = 1
HEADER_ACC_2 = 4
HEADER_GYR = 7
HEADER_MAG = 10

def convert_labels(labels):
    '''
    Labels
    – 1 lying
    – 2 sitting
    – 3 standing
    – 4 walking
    – 5 running
    – 6 cycling
    – 7 Nordic walking
    – 9 watching TV
    – 10 computer work
    – 11 car driving
    – 12 ascending stairs
    – 13 descending stairs
    – 16 vacuum cleaning
    – 17 ironing
    – 18 folding laundry
    – 19 house cleaning
    – 20 playing soccer
    – 24 rope jumping
    – 0 other (transient activities)
    '''
    labels = ['other' if x == 0 else x for x in labels]    
    labels = ['lying' if x == 1 else x for x in labels]        
    labels = ['sitting' if x == 2 else x for x in labels]        
    labels = ['standing' if x == 3 else x for x in labels]        
    labels = ['walking' if x == 4 else x for x in labels]        
    labels = ['running' if x == 5 else x for x in labels]    
    labels = ['cycling' if x == 6 else x for x in labels]        
    labels = ['nordic_walking' if x == 7 else x for x in labels]    

    labels = ['watching_tv' if x == 9 else x for x in labels]   

    labels = ['computer_work' if x == 10 else x for x in labels] 
    labels = ['car_driving' if x == 11 else x for x in labels]   
    labels = ['ascending_stairs' if x == 12 else x for x in labels]    
    labels = ['descending_stairs' if x == 13 else x for x in labels]   

    labels = ['vacuum_cleaning' if x == 16 else x for x in labels]     
    labels = ['ironing' if x == 17 else x for x in labels] 
    labels = ['folding_laundry' if x == 18 else x for x in labels]    
    labels = ['house_cleaning' if x == 19 else x for x in labels]        
    labels = ['playing_soccer' if x == 20 else x for x in labels] 

    labels = ['rope_jumping' if x == 24 else x for x in labels]        

    return labels


from scipy import stats
import pandas as pd

def format(content):
    '''
    Ts(s);Label; BPM;4-20 IMU Hand;21-37 IMU Chest;38-54 IMU Ankle;

    #- The IMU sensory data contains the following columns:  
    #– 1 temperature (°C)  
    #– 2-4 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit  
    #– 5-7 3D-acceleration data (ms-2), scale: ±6g, resolution: 13-bit  
    #– 8-10 3D-gyroscope data (rad/s)  
    #– 11-13 3D-magnetometer data (μT)  
    #– 14-17 orientation (invalid in this data collection) 
    '''
    column_ts = pd.to_numeric(get_col(content, HEADER_TS))
    # stats.describe(column_ts) >> DescribeResult(nobs=8477, minmax=(15.470000000000001, 100.23), mean=57.849999999999994, variance=598.90004999999996, skewness=7.197381257752525e-16, kurtosis=-1.2000000333984933)
    column_label = pd.to_numeric(get_col(content, HEADER_LABEL))
    # 
    column_bpm = pd.to_numeric(get_col(content, HEADER_BPM), errors='ignore')
    # 
    ### HAND IMU
    # acc1 
    column_acc1_x_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_1 ), errors='ignore')
    column_acc1_y_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_1 + 1 ), errors='ignore')
    column_acc1_z_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_1 + 2), errors='ignore')
    # acc2
    column_acc2_x_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_2 ), errors='ignore')
    column_acc2_y_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_2 + 1 ), errors='ignore')
    column_acc2_z_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_ACC_2 + 2), errors='ignore')
    # gyr
    column_gyr_x_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_GYR ), errors='ignore')
    column_gyr_y_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_GYR + 1 ), errors='ignore')
    column_gyr_z_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_GYR + 2), errors='ignore')
    # mag
    column_mag_x_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_MAG ), errors='ignore')
    column_mag_y_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_MAG + 1 ), errors='ignore')
    column_mag_z_hand = pd.to_numeric(get_col(content, HEADER_IMU_HAND + HEADER_MAG + 2), errors='ignore')

    ### CHEST IMU
    column_acc1_x_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_1 ), errors='ignore')
    column_acc1_y_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_1 + 1 ), errors='ignore')
    column_acc1_z_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_1 + 2), errors='ignore')
    # acc2
    column_acc2_x_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_2 ), errors='ignore')
    column_acc2_y_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_2 + 1 ), errors='ignore')
    column_acc2_z_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_ACC_2 + 2), errors='ignore')
    # gyr
    column_gyr_x_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_GYR ), errors='ignore')
    column_gyr_y_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_GYR + 1 ), errors='ignore')
    column_gyr_z_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_GYR + 2), errors='ignore')
    # mag
    column_mag_x_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_MAG ), errors='ignore')
    column_mag_y_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_MAG + 1 ), errors='ignore')
    column_mag_z_chest = pd.to_numeric(get_col(content, HEADER_IMU_CHEST + HEADER_MAG + 2), errors='ignore')

    ### ANKLE IMU
    column_acc1_x_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_1 ), errors='ignore')
    column_acc1_y_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_1 + 1 ), errors='ignore')
    column_acc1_z_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_1 + 2), errors='ignore')
    # acc2
    column_acc2_x_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_2 ), errors='ignore')
    column_acc2_y_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_2 + 1 ), errors='ignore')
    column_acc2_z_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_ACC_2 + 2), errors='ignore')
    # gyr
    column_gyr_x_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_GYR ), errors='ignore')
    column_gyr_y_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_GYR + 1 ), errors='ignore')
    column_gyr_z_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_GYR + 2), errors='ignore')
    # mag
    column_mag_x_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_MAG ), errors='ignore')
    column_mag_y_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_MAG + 1 ), errors='ignore')
    column_mag_z_ankle = pd.to_numeric(get_col(content, HEADER_IMU_ANKLE + HEADER_MAG + 2), errors='ignore')
    ###
    #print("column_acc1_x_ankle:", set(column_acc1_x_hand))
    #print("column_acc1_x_ankle:", set(column_acc1_x_chest))
    #print("column_acc1_x_ankle:", HEADER_IMU_ANKLE + HEADER_MAG + 2)


    # desired format: TS; LABEL; BPM; IMU; POSITION



filepath = "C:\\Users\\paulo\\Documents\\py-har\\datasets\\raw-files\\PAMAP2-2012\\PAMAP2_Dataset\\Protocol\\subject109.dat"
content = [i.strip().split() for i in open(filepath).readlines()]
format(content)
