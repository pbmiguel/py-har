# T1: Training a Model With Pocket Signals and Testing With Arm Signals	


## Analysis of HAR Datasets

*All these datasets are available here:   
https://drive.google.com/drive/folders/1ko3pjX5fUQpCDEMThidhJYHK-sQTgAl_?usp=sharing 

## #1 Realistic Sensor Displacement Benchmark Dataset-2014:RESSDI

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | 3 | 
| Nr of Labels      | 33      | 
| Nr of Columns     | 120      |  
| Header     |   Ts(seconds); Ts(microseconds); ( ACC_X;ACC_Y;ACC_Z;GYR_X;GYR_Y;GYR_Z;MAG_X;MAG_Y;MAG_Z; QUAT_1;QUAT_2;QUAT_3;QUAT_4; ) * 9 BodyPositions ; Label      |
| User Age | 22-37 |
| User Gender | 10M 07F |

## #2 PAMAP2-2012:PAMAP2

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      |  3 | 
| Nr of Labels      | 25 | 
| Nr of Columns     | 54 |  
| Header     |     Ts(s);Label; BPM;4-20 IMU Hand;21-37 IMU Chest;38-54 IMU Ankle;        |
| User Age | 24-30 |
| User Gender | 08M 01F |

*        - The IMU sensory data contains the following columns:  
            – 1 temperature (°C)  
            – 2-4 3D-acceleration data (ms-2), scale: ±16g, resolution: 13-bit  
            – 5-7 3D-acceleration data (ms-2), scale: ±6g, resolution: 13-bit  
            – 8-10 3D-gyroscope data (rad/s)  
            – 11-13 3D-magnetometer data (μT)  
            – 14-17 orientation (invalid in this data collection) 

## #3 Fusion of Smartphone Motion Sensors for Physical Activity Recognition-2014:FUSMPA

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | 5 | 
| Nr of Labels      | 7 | 
| Nr of Columns     | 12 |  
| Header     | TS;Ax;Ay;Az;LAx;LAy;LAz;Gx;Gy;Gz;Mx;My;Mz;Label |
| User Age | 25-30 |
| User Gender | 10M 00F |


## #4 OPPORTUNITY-2012:OPPTY  

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | >5 | 
| Nr of Labels      | 98 | 
| Nr of Columns     | 250 |  
| Header     |  Ts(ms); ( Accelerometer + InertialMeasurementUnit + Location ) * bodyPositions     |
| User Age | ? |
| User Gender | ? |

## #5 Activity recognition with healthy older people using a batteryless wearable sensor-2016:AROLDP

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | 1 | 
| Nr of Labels      | 4 | 
| Nr of Columns     | 9 |  
| Header     | Ts;Ax;Ay;Az;IdAntenna;SignalStrengh;Phase;Frequency;Label |
| User Age | 66-86 |
| User Gender | ? |

## #6 DataSet Daily Log (ADL)-2016:DALYLO
*(con: this dataset requires a lot of pre-processing, since the dataset isn't centralized in one only file but in many)*

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | >=1 | 
| Nr of Labels      | - | 
| Nr of Columns     | - |  
| Header     | - |
| User Age | 22-24 |
| User Gender | 07M 00F |

## #7 Analysis of a Smartphone-Based Architecture with Multiple Mobility Sensors for Fall Detection-2016:ANSAMO
*(pro: we know the gender and age of each subject, thus we can have the walking signals from 2 subjects with more than 50 years old)  

*(con: requires some pre-processing since the dataset displays each sensor values in different tables)  

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      | 5 | 
| Nr of Labels      | 8 | 
| Nr of Columns     | Ts;Acc;Gyro;Magn; |  
| Header     | - |
| User Age | 14-55 |
| User Gender | 11M 06F |

## #8 UniMiB SHAR-2017:UMIBSHAR

*(pro: we know the gender,age,weight, and height of each subject, thus we can have the walking signals from 2 subjects with more than 50 years old)  

*(con: the dataset is stored in matlab files)  

| Characteristic        | Value           |
| ------------- |:-------------:|
| Nr of Positions      |  | 
| Nr of Labels      | 16 | 
| Nr of Columns     | ? |  
| Header     | Accelerometer; Magnitude; Ts |
| User Age | 18-60 |
| User Gender | 06M 24F |
