<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Stargazers][stars-shield]][stars-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
 
# py-har or Human Activity Recognition (HAR) With Python
#### This repository has various use-cases of activity recognition, applying different ML techniques for classifiying tasks.

## First, a survey on activity-recognition datasets
![alt text](https://github.com/pbmiguel/py-har/blob/master/HAR%20Datasets.png)

## The Project's Structure

#### ./Experiments 
has a set of different scenarios that apply different transfer-learning techniques in activity-recognition datasets
### ./Unsupervised Domain Adaptation in HAR
is a collection of projects that apply this technique in HAR
### ./Unsupervised Domain Adaptation
is a collection of projects that apply this technique in different scenarios besides HAR
### ./Working With Datasets
handles the pre-processing of the datasets

## Observation

The construction of the model must take into account the dependency between the collected signals, since they consist in a time-series. 
Therefore avoid typical cross-validation and apply techniques such as forward-chaining, leave-one-subject-out, cross-validation for time-series.  
see more here:  
https://stats.stackexchange.com/questions/14099/using-k-fold-cross-validation-for-time-series-model-selection 
https://robjhyndman.com/hyndsight/tscv/  


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[stars-shield]: https://img.shields.io/github/stars/pbmiguel/py-har?style=flat-square
[stars-url]: https://github.com/pbmiguel/py-har/stargazers
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/paulo-miguel-barbosa/
