# import all the model and package

import argparse
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler as SS
import os

# set the GPU you use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set the data and label parameters
parser = argparse.ArgumentParser(description='train a model to differentiate plasmid and chromosome')
parser.add_argument("matrix", type=str, help="the kmer frequency matrix file in csv format, with header and index")
parser.add_argument("label", type=str, help="the label of matrix")
args = parser.parse_args()

# load your data and label as text format
allmatrix = pd.read_csv(args.matrix, header=0, index_col=0, low_memory=True).values
target = np.loadtxt(args.label)

print("allmatrix shape: {}；label shape: {}".format(allmatrix.shape, target.shape))


# standarize your data
allmatrix = SS().fit_transform(allmatrix)

# transform your data to tensor
x = tf.convert_to_tensor(allmatrix, dtype=tf.float32)
y = tf.convert_to_tensor(target, dtype=tf.int32)

# split train, validation, and test data with ratio 7:1:2
idx = tf.range(allmatrix.shape[0])
idx = tf.random.shuffle(idx)
x_train, y_train = tf.gather(x, idx[:int(0.7 * len(idx))]), tf.gather(y, idx[:int(0.7 * len(idx))])
x_val, y_val = tf.gather(x, idx[int(0.7 * len(idx)):int(0.8 * len(idx))]), tf.gather(y, idx[int(0.7 * len(idx)):int(0.8 * len(idx))])
x_test, y_test = tf.gather(x, idx[int(0.8 * len(idx)):]), tf.gather(y, idx[int(0.8 * len(idx)):])

# set up your batch_size
batchsz = 256

# setup your datasets for training, validation, and test
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz).repeat()

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz)

# check your sample shape
sample = next(iter(db_train))
print(sample[0].shape, sample[1].shape)

# build your neural network model
network = Sequential([layers.Dense(256, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
                      layers.Dropout(0.4), # 0.5 rate to drop

                      layers.Dense(256, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001),),
                      layers.Dropout(0.4),

                      layers.Dense(128, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001),),
                      layers.Dropout(0.5), # 0.5 rate to drop

                      layers.Dense(128, activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0001),),
                      layers.Dropout(0.2),

                      layers.Dense(32, activation='relu'),
                      layers.Dense(10, activation='relu'),
                      layers.Dense(1,activation='sigmoid')])

# build the network and check the network structure
network.build(input_shape=(None, 2080))
network.summary()

# setup eary_stopping and save the model with the best performance
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=80)
checkpoint=ModelCheckpoint('trained_model.h5', monitor='val_accuracy', model='max', verbose=1, save_best_only=True)

# compile the model
network.compile(optimizer=optimizers.Adam(lr=0.001),
               loss=tf.keras.losses.BinaryCrossentropy(), 
               metrics=['accuracy'])

network.fit(db_train, epochs=400, validation_data=ds_val,
            steps_per_epoch=x_train.shape[0]//batchsz,
            validation_steps=2, callbacks=[early_stopping,checkpoint])

network.evaluate(db_test)