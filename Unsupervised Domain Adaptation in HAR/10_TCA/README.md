# OfficeCaltechDomainAdaptation

## What is this ?
This dataset is part of the Computer Vision problematic consisting in making machines learn to detect the presence of an object in an image. Here, we want to learn a classification model that takes as input an image and return the category of the object it contains.

The Office Caltech dataset contains four different domains: amazon, caltech10, dslr and webcam. These domains contain respectively 958, 1123, 295 and 157 images. Each image contains only one object among a list of 10 objects: backpack, bike, calculator, headphones, keyboard, laptop, monitor, mouse, mug and projector.

With this benchmark dataset in Domain Adaptation, we repeatedly take one of the four domains as Source domain S and one of the three remaining as target T. The aim is then to learn to classify images with the data from S to correctly classify the images in T.

## What is available in this repository ?
In addition to the images, we also give features that were extracted from the images to describe them. We give different sets of features that describe all the images in the corresponding folder.

We propose some code in python3 to show how to evaluate the benchmark. What is usually evaluated with this benchmark are Domain Adaptation algorithms. We provide code for a few of them.

## Dependencies
Python3 and some python3 libraries:
 - numpy
 - scipy
 - sklearn

## Example of execution

Program launched by executing the main.py script with python:
```
python3 main.py
```

For each adaptation problem among the 12 possible, each adaptation algorithm chosen at the beginning of the file is applied. Then are reported the mean accuracy and standard deviation. Results (using the default surf [1] features):
```
Feature used:  surf
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT  CORAL
A->C ..........
     23.1  1.9  NA       0.12s
     33.9  2.0  SA       1.03s
     33.3  1.8  TCA     17.49s
     29.8  1.5  OT       1.85s
     28.9  2.5  CORAL    8.95s
A->D ..........
     23.9  2.5  NA       0.02s
     32.1  3.6  SA       0.38s
     28.7  4.7  TCA      0.79s
     40.5  2.5  OT       0.36s
     30.5  3.4  CORAL    8.43s
A->W ..........
     26.5  1.6  NA       0.04s
     31.2  2.4  SA       0.44s
     34.1  2.7  TCA      1.67s
     34.7  2.0  OT       0.46s
     27.2  2.2  CORAL    8.99s
C->A ..........
     22.2  2.8  NA       0.10s
     35.2  2.9  SA       0.80s
     35.9  2.8  TCA     12.38s
     37.4  2.4  OT       1.39s
     32.1  2.4  CORAL    8.88s
C->D ..........
     22.5  4.0  NA       0.02s
     35.2  3.4  SA       0.38s
     34.7  3.4  TCA      0.80s
     44.5  3.7  OT       0.37s
     32.7  2.2  CORAL    9.06s
C->W ..........
     20.2  4.1  NA       0.03s
     30.0  3.2  SA       0.45s
     30.7  4.5  TCA      1.68s
     35.7  5.3  OT       0.51s
     25.9  3.4  CORAL    9.28s
D->A ..........
     26.6  1.8  NA       0.07s
     32.0  1.1  SA       0.66s
     33.4  1.4  TCA      9.19s
     29.2  1.2  OT       0.79s
     29.5  0.7  CORAL    8.49s
D->C ..........
     25.3  1.2  NA       0.08s
     30.6  0.7  SA       0.74s
     31.3  1.4  TCA     13.73s
     29.7  0.9  OT       0.91s
     28.7  0.9  CORAL    8.71s
D->W ..........
     52.2  1.8  NA       0.03s
     79.4  2.0  SA       0.33s
     74.4  2.7  TCA      0.93s
     68.3  2.9  OT       0.32s
     77.9  1.5  CORAL    8.59s
W->A ..........
     23.4  1.0  NA       0.10s
     30.2  1.2  SA       0.81s
     29.5  1.3  TCA     12.26s
     37.3  1.1  OT       1.39s
     28.6  0.8  CORAL    8.70s
W->C ..........
     18.9  1.0  NA       0.12s
     28.7  1.5  SA       0.89s
     29.9  1.0  TCA     17.34s
     34.6  0.8  OT       1.60s
     25.5  0.9  CORAL    8.92s
W->D ..........
     52.0  2.6  NA       0.02s
     83.4  1.9  SA       0.38s
     78.5  1.8  TCA      0.80s
     70.4  1.7  OT       0.34s
     79.7  2.0  CORAL    8.53s

Mean results and total time
     28.1  2.2  NA       0.76s
     40.2  2.2  SA       7.28s
     39.5  2.4  TCA     89.07s
     41.0  2.2  OT      10.30s
     37.3  1.9  CORAL  105.54s
```

By modifying the feature used in the script with CaffeNet [2] features:
```
Feature used:  CaffeNet4096
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT  CORAL
A->C ..........
     70.8  2.7  NA       0.36s
     78.6  2.0  SA       6.55s
     79.6  1.7  TCA     18.43s
     82.3  0.9  OT       4.16s
     76.3  0.8  CORAL  601.37s
A->D ..........
     77.9  3.3  NA       0.08s
     80.7  1.9  SA       3.30s
     86.2  1.9  TCA      0.90s
     93.4  0.7  OT       1.07s
     76.3  2.4  CORAL  592.45s
A->W ..........
     67.1  2.4  NA       0.13s
     80.3  2.5  SA       3.80s
     84.4  2.5  TCA      1.91s
     92.4  1.0  OT       1.52s
     74.9  2.3  CORAL  628.96s
C->A ..........
     81.0  1.9  NA       0.34s
     84.8  1.4  SA       6.29s
     87.3  2.0  TCA     13.89s
     88.5  1.3  OT       3.63s
     81.8  2.1  CORAL  634.86s
C->D ..........
     75.5  5.1  NA       0.09s
     81.1  2.2  SA       3.39s
     83.6  2.9  TCA      0.89s
     93.0  1.5  OT       1.10s
     78.7  1.4  CORAL  620.05s
C->W ..........
     72.4  7.0  NA       0.13s
     76.8  3.0  SA       3.82s
     80.5  2.5  TCA      1.93s
     90.7  1.0  OT       1.51s
     71.3  3.2  CORAL  639.16s
D->A ..........
     70.1  1.8  NA       0.22s
     83.3  1.2  SA       6.14s
     87.3  1.0  TCA      9.96s
     85.9  1.9  OT       2.64s
     82.3  1.4  CORAL  592.33s
D->C ..........
     66.4  1.3  NA       0.25s
     75.2  0.9  SA       6.55s
     78.2  0.9  TCA     14.57s
     79.0  2.5  OT       2.91s
     75.8  0.8  CORAL  599.77s
D->W ..........
     91.7  2.1  NA       0.11s
     96.6  1.1  SA       4.86s
     97.5  1.1  TCA      1.41s
     96.4  0.6  OT       1.25s
     96.4  0.9  CORAL  803.69s
W->A ..........
     69.9  2.0  NA       0.32s
     82.8  0.9  SA       5.85s
     86.7  1.1  TCA     12.77s
     86.8  1.8  OT       3.47s
     77.5  0.7  CORAL  589.03s
W->C ..........
     61.1  2.2  NA       0.35s
     73.4  0.7  SA       6.41s
     76.6  1.3  TCA     18.59s
     77.9  1.8  OT       4.10s
     70.3  0.9  CORAL  602.38s
W->D ..........
     95.9  1.3  NA       0.09s
     99.7  0.4  SA       3.76s
     99.0  1.0  TCA      0.94s
     97.1  0.7  OT       1.15s
     99.6  0.3  CORAL  661.38s

Mean results and total time
     75.0  2.8  NA       2.47s
     82.8  1.5  SA      60.71s
     85.6  1.7  TCA     96.20s
     88.6  1.3  OT      28.50s
     80.1  1.4  CORAL  7565.44s
```

and with GoogleNet [3] features:
```
Feature used:  GoogleNet1024
Number of iterations:  10
Adaptation algorithms used:   NA  SA  TCA  OT  CORAL
A->C ..........
     84.4  1.2  NA       0.14s
     85.7  1.0  SA       1.77s
     87.2  1.1  TCA     20.34s
     90.0  0.7  OT       2.08s
     85.6  1.0  CORAL   18.54s
A->D ..........
     88.4  2.3  NA       0.03s
     87.5  2.9  SA       0.56s
     90.2  3.3  TCA      0.83s
     93.4  0.9  OT       0.51s
     86.2  3.3  CORAL   16.99s
A->W ..........
     82.2  2.3  NA       0.04s
     83.2  1.6  SA       0.64s
     85.9  1.6  TCA      1.75s
     95.7  1.1  OT       0.65s
     82.5  1.2  CORAL   15.97s
C->A ..........
     90.0  1.2  NA       0.11s
     90.9  0.8  SA       1.34s
     92.3  1.4  TCA     13.54s
     93.8  0.4  OT       1.66s
     90.0  0.5  CORAL   18.67s
C->D ..........
     87.3  2.2  NA       0.03s
     88.5  2.1  SA       0.55s
     90.1  2.9  TCA      0.81s
     93.4  1.3  OT       0.50s
     86.1  2.8  CORAL   16.55s
C->W ..........
     84.4  2.4  NA       0.05s
     86.6  1.5  SA       0.68s
     90.9  2.4  TCA      1.74s
     96.9  0.4  OT       0.69s
     83.6  3.3  CORAL   17.15s
D->A ..........
     83.0  1.7  NA       0.08s
     88.0  1.5  SA       1.25s
     90.3  1.5  TCA     10.17s
     91.2  0.9  OT       1.13s
     87.2  1.6  CORAL   18.08s
D->C ..........
     77.2  1.8  NA       0.09s
     83.6  1.1  SA       1.30s
     85.1  1.4  TCA     15.19s
     89.5  0.5  OT       1.32s
     84.7  1.0  CORAL   16.53s
D->W ..........
     97.5  1.1  NA       0.03s
     98.0  0.8  SA       0.48s
     97.3  0.9  TCA      0.94s
     97.8  0.7  OT       0.46s
     98.2  0.7  CORAL   15.97s
W->A ..........
     86.4  1.2  NA       0.11s
     90.0  0.6  SA       1.36s
     92.2  0.6  TCA     13.29s
     92.6  0.5  OT       1.87s
     87.8  0.9  CORAL   17.85s
W->C ..........
     80.0  1.0  NA       0.13s
     84.4  0.7  SA       1.49s
     87.9  0.5  TCA     19.54s
     90.1  0.7  OT       1.96s
     84.7  0.6  CORAL   16.93s
W->D ..........
     99.4  0.3  NA       0.03s
     99.4  0.4  SA       0.54s
     99.8  0.4  TCA      0.82s
     97.3  1.0  OT       0.49s
     99.0  0.6  CORAL   16.22s

Mean results and total time
     86.7  1.5  NA       0.87s
     88.8  1.3  SA      11.97s
     90.8  1.5  TCA     98.98s
     93.5  0.8  OT      13.32s
     88.0  1.5  CORAL  205.45s
```
[1] Gong, B., Grauman, K., & Sha, F. (2014). Learning kernels for unsupervised domain adaptation with applications to visual object recognition. International Journal of Computer Vision, 109(1-2), 3-27.

[2] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., Girshick, R., ... & Darrell, T. (2014, November). Caffe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia (pp. 675-678). ACM.

[3] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
