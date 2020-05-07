# import all the model and package
import argparse
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import pandas as pd
import numpy as np
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the data parameters
parser = argparse.ArgumentParser(description='using the model to predict a sequence origin with kmer frequency, this'
                                             'the model is used for 6mer matrix')
parser.add_argument("matrix", type=str, help="the kmer frequency matrix file in csv format, with header and index")
args = parser.parse_args()

# load your data
allmatrix = pd.read_csv(args.matrix, header=0, index_col=0, low_memory=True).values

# scaler your data
scaler = pickle.load(open('scaler.pkl', 'rb'))
allmatrix = scaler.transform(allmatrix)

print("allmatrix shape: {}".format(allmatrix.shape))

# transform your data to tensor
x = tf.convert_to_tensor(allmatrix)

# load the model for prediction
network = tf.keras.models.load_model('5k6mer_model.h5', compile=True)

# predict your data
prediction=network.predict(x)
print(prediction)

# save the prediction result
np.savetxt("prediction_result.txt", prediction, fmt="%.4f")
