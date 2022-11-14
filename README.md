# Plasmid_ML
This repository contains all the code to build a Machine learning model to differentiate plasmid and chromosome.

This model is used to predict a sequence origin (plasmid or chromosome) based on the kmer frequency. Typically, we used a 5000 bp sequence and calculate its 6mer frequency, which produces 2080 kmer features. The frequency matrix is fed into the model to predict the sequence origin.

The ID info of plasmid and chromosome (with length >= 5000bp) we used to established the model are listed, namely plasmid_id_5k and plasmid_id_5k in the repo, respectively. All the sequences were downloaded from https://www.bv-brc.org. The three columns of two files are GeneBank Accession NO,Genome Name,and Genome ID in the bv-brc database.


## Prerequisites
The model requires the installation of Python 3.8, scikit-learn 0.21.3, tensorflow 2.0, pandas 0.24.2, NumPy 1.16.0, and the KMC 3.1.1 tool. KMC3 can be installed based on manual [KMC3](http://sun.aei.polsl.pl/REFRESH/index.php?page=projects&project=kmc&subpage=download). The model could be run on a laptop with 2G gpu, and a tensorflow 2.0 or above should be installed with gpu.

## Installation
```
git clone https://github.com/Xiaohui-Z/Plasmid_ML.git
```

## Features order

We used KMC3 to generate 6mer frequency for each sequence fragment. The “kmer_order.csv” file showed the order of 6mer type KMC3 produced (in alphabetical order), which is also the order of feature for the input CSV that used for model prediction.  


## usage
model_predict.py is the script which use our model to predict the origin of a 5000bp DNA fragment. Model is established on Tensorflow 2.0, which use 6mer frequecy produced by a 5000bp DNA fragment to predict its origin (Plasmid or Chromosome). 

A DNA sequence should be chopped to 5000bp and then KMC3 is used to calculate the 6mer frequency. Each row represents a sequence and each column means a 6mer type. The script model_predict.py can be used to classify the sequences based on the frequency of 6mer, containing 2080 features. The data should be csv format with header and index column. An example to predict on the testdata.csv is below. Generally, prediction of 100 samples cost less than 5 minutes. 

```
python model_predict.py testdata.csv
```
The testresult was stored in a file called "prediction_result.txt", which showed the probability that one fragment was predicted as plasmid by the model.

## Training new models
We also provide a script used to train your own model. For example, `training_data.csv` is your training matrix in csv format (with header and index column), and `training_label.txt` is your label in txt format. Usually, training a dataset containing 10,000 samples cost about 10 minutes.

To trained a model with your own sequence data:

1.use KMC3 to calculate 6-mer frequency of each sequence.
```
~/bin/KMC3bX/kmc -k6 -ci1 -fa -cs5000 yourfastafile.fa interfile temp/
~/bin/KMC3bX/kmc_dump interfile kmerfile
```
2.merge the 6mer frequency of each sequence to a matrix with order in the "kmer_order.csv" file, which get the traning_data. Label the plasmid as 1 and chromosome as 0 to formed the label file. 

3.training your model 

This command will train your data and get a new model named "trained_model.h5" in your working directory:
```
python model_train.py training_data.csv training_label.txt
```
