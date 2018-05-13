import pandas as pd
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
        self.data = train

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