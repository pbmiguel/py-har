### Read File
import scipy.io as sio

def full_data():
    mat_contents = sio.loadmat('C:\\Users\\paulo\\Documents\\py-har\\datasets\\UMIBSHAR-2017\\data\\full_data.mat')
    print(type(mat_contents))

    for key in mat_contents:
        print(key)

    header = mat_contents['__header__']
    version = mat_contents['__version__']
    globals = mat_contents['__globals__']
    data = mat_contents['full_data']

    print(type(header), header)
    print(type(version), version)
    print(type(globals), globals)
    print(type(data))
    print(len(data[0]), len(data[0][0]), len(data[0][0][0]), len(data[0][0][0][0]), len(data[0][0][0][0][0]), len(data[0][0][0][0][0][0]))
    # >> 5 1 1 17 2 1
    # Each activity record is made of 6 rows: the first three contain acceleration data along  x,y and z directions, the forth and fitfth row, the time instants and the sixth row the magnitudo of the raw signal. 

    # 8x array 
    # print(data[0][0][0][0][0][0][0][0])

    #print(len(data[0][0][0][0]))


def adl_data():
    mat_contents = sio.loadmat('C:\\Users\\paulo\\Documents\\py-har\\datasets\\UMIBSHAR-2017\\data\\adl_data.mat')
    print(type(mat_contents))

    for key in mat_contents:
        print(key)

    header = mat_contents['__header__']
    version = mat_contents['__version__']
    globals = mat_contents['__globals__']
    data = mat_contents['adl_data']
    print(len(data), len(data[2]))
    # >> 7579 x 453 
    # raw data: the 453-dimensional patterns obtained by concatenating the 151 acceleration values recorded along each Cartesian direction;

def acc_data():
    mat_contents = sio.loadmat('C:\\Users\\paulo\\Documents\\py-har\\datasets\\UMIBSHAR-2017\\data\\acc_data.mat')
    print(type(mat_contents))

    for key in mat_contents:
        print(key)

    header = mat_contents['__header__']
    version = mat_contents['__version__']
    globals = mat_contents['__globals__']
    data = mat_contents['acc_data']
    print(len(data), len(data[2]))
    # >> 11771 x 453 
    # raw data: the 453-dimensional patterns obtained by concatenating the 151 acceleration values recorded along each Cartesian direction;
    # The dataset contains a total of 11,771 samples describing both activities of daily living (7579) and falls (4192)

acc_data()