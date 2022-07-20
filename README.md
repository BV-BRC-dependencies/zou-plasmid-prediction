# Plasmid_ML
This repository contains all the code to build a Machine learning model to differentiate plasmid and chromosome.

This model is used to predict a sequence origin (plasmid or chromosome) based on the kmer frequency. Typically, we used a 5000 bp sequence and calculate its 6mer frequency, which produces 2080 kmer features. The frequency matrix is fed into the model to predict the sequence origin.

## Prerequisites
The model requires the installation of Python 3.8, tensorflow 2.0, and the KMC 3.1.1 tool. KMC3 can be installed based on manual [KMC3](http://sun.aei.polsl.pl/REFRESH/index.php?page=projects&project=kmc&subpage=download)

## Installation
```
git clone https://github.com/Xiaohui-Z/Plasmid_ML.git
```

## usage
A DNA sequence should be chopped to 5000bp and then KMC3 is used to calculate the 6mer frequency. Each row represents a sequence and each column means a 6mer type. The script model_predict.py can be used to classify the sequences based on the frequency of 6mer, containing 2080 features. The data should be csv format with header and index column. An example to predict on the testdata.csv is below

```
python model_predict.py testdata.csv
```
The testresult was stored in a file called "prediction_result.txt", which showed the probability that one fragment was predicted as plasmid by the model.

## Training new models
We also provide a script used to train your own model. For example, `training_data.csv` is your training matrix in csv format (with header and index column), and `training_label.txt` is your label in txt format.

This command will train your data and get a new model named "trained_model.h5" in your working directory:
```
python model_train.py model_totaldata.csv model_totaltarget.txt
```
