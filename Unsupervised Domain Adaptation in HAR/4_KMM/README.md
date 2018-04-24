# KMM
Kernel Mean Matching (KMM) is a well-known method for bias correction(training 
and test data are drawn from different distributions.) by estimating 
density ratio between training and test data distribution. This
mechanism re- weights training data instances so that their weighted data
distribution resembles that of the observed test data distribution.


KMM source code:
https://github.com/swarupchandra/multistream/blob/master/kmm.py


To Run:
select one pair of tweet document as train and test.
put those data files and the python code into one folder in Pycharm
Run "python KMM.py train.arff test.arff"

The output is
"
the accuracy without KMM87.5
the accuracy with KMM88.75
