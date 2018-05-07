
mypath = "./individual-data"
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

'''
###################################
### verify if columns are the same
###################################
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
output_all = pd.DataFrame()
i = 0
for filename in files:
    content = pd.read_csv(mypath + "\\" + filename , sep=';')
    print(filename)
    cols = content.columns
    hashes = dict()
    for col in cols:
        hash_col = hash(content[col].to_string())
        if hashes.get(hash_col) != None:
            col2 = hashes.get(hash_col)
            raise Exception("Same Columns!", col, "and", col2);
        hashes[hash_col] = col
'''
'''
    >>> orientation columnas are equal
'''
'''
########################################
### test correlation between columns ###
########################################

files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
output_all = pd.DataFrame()
i = 0
for filename in files:
    content = pd.read_csv(mypath + "/" + filename , sep=';')
    corr_matrix = content.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    #
    print("corr > 0.95")
    for upper_col in upper.columns:
        #print(upper_col)
        #print(type(upper[upper_col]))        
        i = 0
        for row in upper[upper_col]:
            #print("row:", row)
            if row > 0.95:
                print(filename, upper_col, upper.columns[i], row)
            i+=1
        #print(upper_col , "\n", type(corr_matrix[upper_col]))
'''
'''
>>> corr > 0.95
    subject101.csv acc2_x acc1_x 0.971734449551
    subject101.csv ts Unnamed: 0 1.00000000001
    subject102.csv acc2_x acc1_x 0.982795014689
    subject102.csv acc2_z acc1_z 0.95460077337
    subject102.csv ts Unnamed: 0 0.999999999999
    subject103.csv acc2_x acc1_x 0.991285220113
    subject103.csv acc2_y acc1_y 0.959772680632
    subject103.csv acc2_z acc1_z 0.951923513697
    subject103.csv ts Unnamed: 0 1.0
    subject104.csv acc2_x acc1_x 0.994870315719
    subject104.csv acc2_y acc1_y 0.964496265401
    subject104.csv acc2_z acc1_z 0.976505095364
    subject104.csv ts Unnamed: 0 1.00000000001
    subject105.csv acc2_x acc1_x 0.975445686093
    subject105.csv acc2_z acc1_z 0.956698742529
    subject105.csv ts Unnamed: 0 1.00000000001
    subject106.csv acc2_x acc1_x 0.956953081452
    subject106.csv ts Unnamed: 0 1.00000000001
    subject107.csv acc2_x acc1_x 0.96922255608
    subject107.csv ts Unnamed: 0 1.0
    subject108.csv acc2_x acc1_x 0.962128894491
    subject108.csv ts Unnamed: 0 1.00000000001
    subject109.csv bpm Unnamed: 0 0.95193096068
    subject109.csv ts Unnamed: 0 1.0
    subject109.csv ts bpm 0.95193096068
'''