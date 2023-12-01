import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional,Embedding
import re

import json
# cargar datos del json
# Loading json data
with open('data/data_full.json') as file:
  data = json.loads(file.read())

# Loading out-of-scope intent data
val_oos = np.array(data['oos_val'])
train_oos = np.array(data['oos_train'])
test_oos = np.array(data['oos_test'])

# Loading other intents data
val_others = np.array(data['val'])
train_others = np.array(data['train'])
test_others = np.array(data['test'])

# Merging out-of-scope and other intent data
val = np.concatenate([val_oos,val_others])
train = np.concatenate([train_oos,train_others])
test = np.concatenate([test_oos,test_others])
data = np.concatenate([train,test,val])
data = data.T

text = data[0]
labels = data[1]

train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)

# Convertir el texto en una lista de palabras
words = []
for txt in text:
    words += text_to_word_sequence(txt)

# Contar el número de palabras únicas
unique_words = set(words)
num_unique_words = len(unique_words)

# Imprimir el número de palabras únicas
print("Número de palabras únicas: ", num_unique_words)

max_num_words = 14000 #el numero de palabras unicas es de 211 entonces agregamos un maximo con holgura
classes = np.unique(labels)

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(train_txt)
word_index = tokenizer.word_index

word_index

ls=[]
for c in train_txt:
    ls.append(len(c.split()))   #cada consulta se convierte en una lista de palabras y se cuenta el numero de palabras

maxLen=int(np.percentile(ls, 98))   #se calcula el percentil 98 de la lista de palabras

train_sequences = tokenizer.texts_to_sequences(train_txt)   #convierte las consultas en secuencias de tokens
print("train sequence tokenice:\n",train_sequences)

train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')     #rellena con ceros las secuencias para que todas tengan la misma longitud dada por el percentil 98
print("train sequence pad sequence:\n",train_sequences)

test_sequences = tokenizer.texts_to_sequences(test_txt)
print("test sequence tokenice:\n",test_sequences)
test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')
print("test sequence tokenice:\n",test_sequences)

label_encoder = LabelEncoder() # codifica las clases en numeros
integer_encoded = label_encoder.fit_transform(classes) # codifica las clases en numeros

print("Clases: ", classes)
print("Clases codificadas: ", integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

print(integer_encoded)

onehot_encoder.fit(integer_encoded)

print("Onehot catgories:",onehot_encoder.categories_) # codifica las clases en numeros

# dimesiones del onehot_encoder
num_classes = len(onehot_encoder.categories_[0])
print("Número de clases: ", num_classes)
print("Clases \n", onehot_encoder.categories_[0])

train_label_encoded = label_encoder.transform(train_label) # codifica las clases en numeros

train_label_encoded

train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)

train_label = onehot_encoder.transform(train_label_encoded)    # codifica las clases en numeros

#ahora lo mismo pero para los datos de testeo

test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)

print("pre-onehot:\n",test_labels)

test_labels = onehot_encoder.transform(test_labels_encoded)

print("\n\npos-onehot:\n",test_labels)

w2v_model = Word2Vec(sentences=[words], vector_size=100, window=5, min_count=1, workers=4)

modelo_LSTM = Sequential()
modelo_LSTM.add(Embedding(input_dim=num_unique_words, output_dim=100, input_length=train_sequences.shape[1], trainable=False, weights=[w2v_model.wv.vectors]))
modelo_LSTM.add(Bidirectional(LSTM(256, return_sequences=True), 'concat')) #Probablemente pueda usar 71, segun el calculo de la formula: 2/3*(n_input+n_output)
modelo_LSTM.add(Dropout(0.2))
modelo_LSTM.add(LSTM(256, return_sequences=False))
modelo_LSTM.add(Dropout(0.2)) # desactiva el 20% de las neuronas
modelo_LSTM.add(Dense(50, activation='relu'))
modelo_LSTM.add(Dense(151, activation='softmax')) # Esta es la capa de salida, va a tener tantas neuronas como clases haya
modelo_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
h = modelo_LSTM.fit(train_sequences, train_label, epochs=10, batch_size=64, shuffle=True, validation_data=(test_sequences, test_labels))