'''
    headers:
        - 120 columns
        Timestamp (second) | Timestamp (microsecond ) | RLA | RUA | BACK | LUA | LLA | RC | RT | LT | LC | Label
        sensors:        ACC : X ACC : Y ACC : Z 
                        GYR : X GYR : Y GYR : Z 
                        MAG : X MAG : Y MAG : Z
                        QUAT : 1 QUAT : 2 QUAT : 3 QUAT : 4

    labels (33):  
            // The dataset consists of a set of typical warm up, fitness and cool down exercises which are
            summarized in Table 3). We have included whole body (L1-L3), (L5-L8), (L31-33) as well
            as body part specific activities focused on the trunk (L9-L18), upper extremities (L19-L25)
            and lower extremities (L26-L29).

            L1: Walking (1 min) L12: Waist rotation (20x) L23: Shoulders high amplitude rotation (20x)
            L2: Jogging (1 min) L13: Waist bends (reach foot with opposite hand) (20x) L24: Shoulders low amplitude rotation (20x)
            L3: Running (1 min) L14: Reach heels backwards (20x) L25: Arms inner rotation (20x)
            L4: Jump up (20x) L15: Lateral bend (10x to the left + 10x to the right) L26: Knees (alternatively) to the breast (20x)
            L5: Jump front & back (20x) L16: Lateral bend arm up (10x to the left + 10x to the right) L27: Heels (alternatively) to the backside (20x)
            L6: Jump sideways (20x) L17: Repetitive forward stretching (20x) L28: Knees bending (crouching) (20x)
            L7: Jump leg/arms open/closed (20x) L18: Upper trunk and lower body opposite twist (20x) L29: Knees (alternatively) bend forward (20x)
            L8: Jump rope (20x) L19: Arms lateral elevation (20x) L30: Rotation on the knees (20x)
            L9: Trunk twist (arms outstretched) (20x) L20: Arms frontal elevation (20x) L31: Rowing (1 min)
            L10: Trunk twist (elbows bended) (20x) L21: Frontal hand claps (20x) L32: Elliptic bike (1 min)
            L11: Waist bends forward (20x) L22: Arms frontal crossing (20x) L33: Cycling (1 min)
    
    positions (9):
            LC: Left calf
            RC: Right calf
            LT: Left thigh
            RT: Right thigh
            LLA: Left lower arm
            RLA: Right lower arm
            LUA: Left upper arm
            RUA: Right upper arm
            BACK: Back

    experiments:
        - Ideal-placement or default scenario: The sensors are positioned by the instructor to predefined
        locations within each body part. The data stemming from this scenario could be
        considered as the training set for supervised activity recognition systems.

        - Self-placement scenario: The user is asked to position 3 sensors himself on the body part
        specified by the instructor. This scenario tries to simulate some of the variability that may
        occur in the day to day usage of an activity recognition system, involving wearable or selfattached
        sensors. Normally the self-placement will lead to on-body sensor setups that differ
        with respect to the ideal- placement. Nevertheless, this difference may be minimal if the
        subject places the sensor close to the ideal position.

        - Mutual-displacement: An intentional de-positioning of sensors using rotations and translations
        with respect to the ideal placement is introduced by the instructor. One of the key interests
        of including this last scenario is to investigate how the performance of a certain method
        degrades as the system drifts far from the initial setup. The number of sensors displaced in
        this scenario increases from 4 to 7.
'''
HEADER_TS_S = 0
HEADER_TS_MS = 1
HEADER_LABEL = 119

HEADER_RLA = 2
HEADER_RUA = 15
HEADER_BACK = 28
HEADER_LUA = 41
HEADER_LLA = 54
HEADER_RC = 67
HEADER_RT = 80
HEADER_LT = 93
HEADER_LC = 106

HEADER_ACC = 0
HEADER_GYR = 3
HEADER_MAG = 6
HEADER_QUAT = 9


def transform_label(label):
     return {
        0: None,
        1: 'walking',
        2: 'jogging',
        3: 'running',
        4: 'jump_up',
        5: 'jump_front_and_back',

        6: 'jump_sideways',
        7: 'jump_leg_arms_open',
        8: 'jump_rope',
        9: 'trunk_twist_arms_outstretched',
        10: 'trunk_twist_elbows_bended',

        11: 'waist_bends',
        12: 'waist_rotation',
        13: 'waist_bends',
        14: 'reach_heels_backwards',
        15: 'lateral_bend',

        16: 'lateral_bend_arm_up',
        17: 'stretching_forward',
        18: 'uppter_trunk_and_lower_body_opposite_twist',
        19: 'arms_lateral_elevation',
        20: 'arms_frontal_elevation',

        21: 'frontal_hands_clap',
        22: 'arms_frontal_crossing',
        23: 'shoulders_high_amplitude_rotation',
        24: 'shoulders_low_amplitude_rotation',
        25: 'arms_innter_rotation',

        26: 'knees_to_the_breast',
        27: 'heels_to_the_backside',
        28: 'knees_bending_crouching',
        29: 'knees_bending_forward',
        30: 'rotation_on_the_knees',

        31: 'rowing',
        32: 'elliptic_bike',
        33: 'cycling',

    }.get(label, None) 

import pandas as pd
import numpy

def read_file(subject, experiment, filename):
    with open(filename) as f:
        f = f.readlines()

    output = pd.DataFrame()
    print(len(f))
    i = 0
    #
    # Creates a list containing 5 lists, each of 8 items, all set to 0
    h, w = len(f), 20;
    Matrix = [[-1 for x in range(w)] for y in range(h)] 
    #
    if len(f) <= 0:
        print("empty")
        return
    
    for line in f:
        i+=1
        cols = line.split('\t')
        #
        if i % 50000 == 0: print(i)
        #
        ts_s = cols[HEADER_TS_S]
        ts_ms = cols[HEADER_TS_MS]
        ts_s_ms = float(ts_s) + float(ts_ms) / 1000000
        label_tmp = cols[HEADER_LABEL]
        label = transform_label(float(label_tmp))
        #print(label_tmp)
        if label is not None:
            for pos, position_name in [(HEADER_RLA, 'right_lower_arm'), (HEADER_RUA, 'right_upper_arm'), (HEADER_BACK, 'back'), (HEADER_LUA, 'left_upper_arm'),  (HEADER_LLA, 'left_lower_arm'),  (HEADER_RC, 'right_calf'),  (HEADER_RT, 'right_thigh'),  (HEADER_LT, 'left_thigh'),  (HEADER_LC, 'left_calf')]:
                #print(pos, position_name)
                # acc
                acc_x = cols[pos + HEADER_ACC]
                acc_y = cols[pos + HEADER_ACC + 1 ]
                acc_z = cols[pos + HEADER_ACC + 2 ]
                # gyr
                gyr_x = cols[pos + HEADER_GYR]
                gyr_y = cols[pos + HEADER_GYR + 1 ]
                gyr_z = cols[pos + HEADER_GYR + 2 ]
                # mag
                mag_x = cols[pos + HEADER_MAG]
                mag_y = cols[pos + HEADER_MAG + 1 ]
                mag_z = cols[pos + HEADER_MAG + 2 ]
                # quaternion
                quat_1 = cols[pos + HEADER_QUAT]
                quat_2 = cols[pos + HEADER_QUAT + 1 ]
                quat_3 = cols[pos + HEADER_QUAT + 2 ]
                quat_4 = cols[pos + HEADER_QUAT + 3 ]
                #

                Matrix[i][0] = ts_s
                Matrix[i][1] = ts_ms
                Matrix[i][2] = ts_s_ms
                Matrix[i][3] = position_name
                Matrix[i][4] = label
                Matrix[i][5] = subject     
                Matrix[i][6] = experiment     


                Matrix[i][7] = acc_x
                Matrix[i][8] = acc_y
                Matrix[i][9] = acc_z

                Matrix[i][10] = gyr_x
                Matrix[i][11] = gyr_y
                Matrix[i][12] = gyr_z

                Matrix[i][13] = mag_x
                Matrix[i][14] = mag_y
                Matrix[i][15] = mag_z

                Matrix[i][16] = quat_1
                Matrix[i][17] = quat_2
                Matrix[i][18] = quat_3
                Matrix[i][19] = quat_4

                # print(quat_1)
                '''output = output.append(pd.DataFrame([{
                    'ts_s': ts_s,
                    'ts_ms': ts_ms,
                    'ts_s_ms': ts_s_ms,
                    'position': position_name,
                    'label': label,
                    'subject': subject,
                    'experiment': experiment,

                    'acc_x': acc_x,
                    'acc_y': acc_y,
                    'acc_z': acc_z,

                    'gyr_x': gyr_x,
                    'gyr_y': gyr_y,
                    'gyr_z': gyr_z,

                    'mag_x': mag_x,
                    'mag_y': mag_y,
                    'mag_z': mag_z,

                    'quat_1': quat_1,
                    'quat_2': quat_2,
                    'quat_3': quat_3,            
                    'quat_4': quat_4,            
                }]))'''
    #print(len(f), len(output))
    #output.to_csv(subject +"_"+ experiment + ".csv", sep=';')
    #clean_matrix = [[x for x in Matrix] for y in range(h)] 
    a = numpy.asarray(clean_matrix)
    numpy.savetxt( str(subject + "_" + experiment + ".csv" ), a, delimiter=";", fmt='%s')
    return output

def read_log_files():
    mypath = "C:\\Users\\paulo\\Documents\\py-har\\datasets\\RESSDI-2014\\data"
    from os import listdir
    from os.path import isfile, join

    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    i = 0
    for filename in files:
        try:
            msg_1 = str(filename).split('.')[0]
            subject = msg_1.split("_")[0]
            experiment = msg_1.split('_')[1]
            #if subject not in ['subject10', 'subject11', 'subject12', 'subject13', 'subject14']:    
            print(subject, experiment)
            out = read_file(subject, experiment, mypath + "\\" + filename)
            #if i == 0: 
            #    out.to_csv('all.csv', sep=';')
            #else:
            #    out.to_csv('all.csv', mode='a', header=False, sep=';')
            i+=1
        except Exception as e:
            print(e)
            #traceback.print_exc()
            pass

read_log_files()


#if label is not None:
    # add to dataframe
# the label is most of the time at 0! 