import pandas as pd
PATH  = './datasets/ANSAMO/'
from labels import LABELS

class ANSAMO( object ):

    data = 0

    def __init__(self, window='3000'):
        train = pd.read_csv(PATH + 'all_values_with_fe_'+ window + '_ms.csv')
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
        cols_to_remove = ['Unnamed: 0', 'Fts(ms)', 'Its(ms)', 'accX', 'accY', 'accZ',
       'experiment', 'filename', 'gyrX', 'gyrY', 'gyrZ', 'magX',
       'magY', 'magZ', 'position', 'userAge', 'userGender', 'userID',]
        dataframe = dataframe.drop(cols_to_remove, axis=1)
        return dataframe
    
    def replace_labels(self, dataframe):
        dataframe['label'] =  dataframe['label'].replace(['GoDownstairs'], LABELS.STAIRS_DOWN)
        dataframe['label'] =  dataframe['label'].replace(['GoUpstairs'], LABELS.STAIRS_UP)
        dataframe['label'] =  dataframe['label'].replace(['Jogging'], LABELS.RUNNING)
        dataframe['label'] =  dataframe['label'].replace(['Sitting'], LABELS.SITTING)
        dataframe['label'] =  dataframe['label'].replace(['Hopping'], LABELS.HOPPING)
        dataframe['label'] =  dataframe['label'].replace(['Walking'], LABELS.WALKING)
        dataframe['label'] =  dataframe['label'].replace(['Bending'], LABELS.BENDING)

        return dataframe