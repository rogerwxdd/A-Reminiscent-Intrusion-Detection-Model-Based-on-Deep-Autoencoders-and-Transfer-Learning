from IPython import embed

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Activation, Dense
from numpy import average
from numpy import array
from keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import sklearn

EPOCHS = 5
NWORKERS = 5
EPOCHS_WORKER = 50

def fedAvg(members):
  n_layers = len(members[0].get_weights())
  weights = [1.0/len(members) for i in range(1, len(members)+1)]
  avg_model_weights = list()
  for layer in range(n_layers):
    layer_weights = array([model.get_weights()[layer] for model in members])
    avg_layer_weights = average(layer_weights, axis=0, weights=weights)
    avg_model_weights.append(avg_layer_weights)
  model = clone_model(members[0])
  model.set_weights(avg_model_weights)
  model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return model

def geraNModelosCopiaFederado(modeloFederado, n):
  modelos = [clone_model(modelFederado) for i in range(n)]
  for modelo in modelos:
    modelo.set_weights(modelFederado.get_weights())
    modelo.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  return modelos

dataView = 'ORUNADA'

caminho_dataset = 'dataset/'
dataset = ['01','02','03','04','05','06','07','08','09','10','11']

for data in dataset:

  print('\nIniciado: ' + data)

  df = pd.read_csv(caminho_dataset + 'ORUNADA2014_B' + data + '.csv')
  df = df.sample(frac=1).reset_index(drop=True)
  x = df.drop('class', axis=1)
  inputs = np.asarray(x)

  inputs = MinMaxScaler().fit_transform(inputs)
  labels = np.asarray(df['class'])
  labels = to_categorical(labels)

  lista_dataset_x = []
  lista_dataset_y = []
  lista_workers = []
  lista_dataset_federado = []

  for i in range(NWORKERS):
    inicio = i * int(len(inputs) / NWORKERS)
    fim = (i+1) * int(len(inputs) / NWORKERS)

    x = inputs[inicio:fim]
    y = labels[inicio:fim]
    lista_dataset_x.append(x)
    lista_dataset_y.append(y)

  modelFederado = Sequential()
  modelFederado.add(Dense(500, activation='relu', input_dim=numDataView))
  modelFederado.add(Dense(100, activation='relu'))
  modelFederado.add(Dense(50, activation='relu'))
  modelFederado.add(Dense(2, activation='softmax'))
  modelos = geraNModelosCopiaFederado(modelFederado, NWORKERS)

  for i in range(1, EPOCHS + 1):
    print('Epoca [{:2d}/{:2d}]'.format(i, EPOCHS))
    for j in range(len(modelos)):
      history = modelos[j].fit(lista_dataset_x[j], lista_dataset_y[j], epochs=EPOCHS_WORKER, verbose=0)
      print('\tTreinou modelo: [{:2d}/{:2d}] da epoca [{:2d}] (acc {:.6f} \tLoss: {:.6f})'.format(j, len(modelos), i, history.history['accuracy'][0], history.history['loss'][0]))
    modelFederado = fedAvg(modelos)
    modelos = geraNModelosCopiaFederado(modelFederado, NWORKERS)

  modelFederado.save('saveModel/ORUNADA2014_B' + str(data))

