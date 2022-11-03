# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

from scipy import stats
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import confusion_matrix
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
def create_dataset(time_steps, step,train=True):
    global enc
    x_list, y_list = [], []
    for i in range(0, len(r) - time_steps, step):
        v = r.iloc[i:(i + time_steps)]
        labels = v.pop('label')
        x_list.append(v.values)
        y_list.append(stats.mode(labels)[0][0])
    x_train = np.array(x_list)
    y_train = np.array(y_list).reshape(-1, 1)
    x_train.shape

    if(train):
        enc = enc.fit(y_train)    # Call 'fit' with appropriate arguments before using this estimator.
    y_train = enc.transform(y_train)
    return (x_train,y_train)

def create_model(trainX, trainy):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # evaluate model
    return model

def evaluate_model(model, testX, testy, batch_size, verbose):

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

r = pd.read_feather("respeck.feather")
r.drop(columns=['timestamp','activity_type'], axis=1, inplace=True)
time_steps = 50
step = 10
x_train,y_train = create_dataset(time_steps, step)

model = create_model(trainX=x_train,trainy=y_train)

model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

rt = pd.read_feather("respeck2.feather")
rt.drop(columns=['timestamp','activity_type'], axis=1, inplace=True)
time_steps = 50
step = 10
x_test,y_test = create_dataset(time_steps, step)
# acc=evaluate_model(model,x_test,y_test,batch_size=32,verbose =0)
# print(acc)
predy = model.predict(x_test)


def plot_cm(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(12, 10)) 
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f", 
        cmap=sns.diverging_palette(220, 20, n=7),
        ax=ax
    )

    plt.xticks(rotation=70)
    plt.yticks(rotation=90)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names, rotation=0)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # ta-da!

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1)

rcParams['figure.figsize'] = 22, 10

plot_cm(
    enc.inverse_transform(y_test),
    enc.inverse_transform(predy),
    enc.categories_[0]
) 
