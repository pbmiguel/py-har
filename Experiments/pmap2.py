import pandas as pd
from labels import LABELS

PATH  = './datasets/PMAP2/'

class PMAP2( object ):

    data = 0

    def __init__(self, window='3'):
        train = pd.read_csv(PATH + window + 's_subject101.csv', sep=';')
        train = train.append( pd.read_csv(PATH + window + 's_subject102.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject103.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject104.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject105.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject106.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject107.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject108.csv', sep=';'))
        train = train.append( pd.read_csv(PATH + window + 's_subject109.csv', sep=';'))
        self.data = self.replace_labels(train)

    def get_data(self, positions=None, labels=None, users=None):
        out = self.data
        if positions is not None:
            out = out[out['position'].isin(positions)]
        if labels is not None:
            out = out[out['label'].isin(labels)]
        if users is not None:
            out = out[out['userID'].isin(users)]
        return self.remove_cols(out)

    def get_positions(self, users=None, labels=None):
        out = self.data
        if labels is not None:
            out = out[out['label'].isin(labels)]
        if users is not None:
            out = out[out['userID'].isin(users)]
        return set(out['position'])    
    
    
    def get_labels(self, users=None, positions=None):
        out = self.data
        if positions is not None:
            out = out[out['position'].isin(positions)]
        if users is not None:
            out = out[out['userID'].isin(users)]
        return set(out['label']) 

    
    def remove_cols(self, dataframe):
        cols_to_remove = ['Unnamed: 0', 'accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ',
    'magX', 'magY', 'magZ', 'position', 'ts' , 'userID']
        dataframe = dataframe.drop(cols_to_remove, axis=1)
        return dataframe
    
    def replace_labels(self, dataframe):
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
        dataframe['label'] =  dataframe['label'].replace([0], LABELS.OTHER)
        dataframe['label'] =  dataframe['label'].replace([1], LABELS.LYING)
        dataframe['label'] =  dataframe['label'].replace([2], LABELS.SITTING)
        dataframe['label'] =  dataframe['label'].replace([3], LABELS.STANDING)
        dataframe['label'] =  dataframe['label'].replace([4], LABELS.WALKING)
        dataframe['label'] =  dataframe['label'].replace([5], LABELS.RUNNING)
        #
        dataframe['label'] =  dataframe['label'].replace([6], LABELS.CYCLING)
        dataframe['label'] =  dataframe['label'].replace([7], LABELS.NORDIC_WALKING)
        dataframe['label'] =  dataframe['label'].replace([9], LABELS.TV)
        dataframe['label'] =  dataframe['label'].replace([10], LABELS.COMPUTER)
        dataframe['label'] =  dataframe['label'].replace([11], LABELS.CAR)
        #
        dataframe['label'] =  dataframe['label'].replace([12], LABELS.STAIRS_UP)
        dataframe['label'] =  dataframe['label'].replace([13], LABELS.STAIRS_DOWN)
        dataframe['label'] =  dataframe['label'].replace([16], LABELS.VACCUM)
        dataframe['label'] =  dataframe['label'].replace([17], LABELS.IRONING)
        dataframe['label'] =  dataframe['label'].replace([18], LABELS.LAUNDRY)
        #
        dataframe['label'] =  dataframe['label'].replace([19], LABELS.HOUSE_CLEANING)
        dataframe['label'] =  dataframe['label'].replace([20], LABELS.SOCCER)
        dataframe['label'] =  dataframe['label'].replace([24], LABELS.ROPE_JUMPING)
        return dataframe
