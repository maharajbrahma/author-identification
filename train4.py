from time import time
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten,Dropout
from keras.layers import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

path = 'Dataset/'
texts = []
labels = []
label_index= {}
authors = sorted(os.listdir(path))
print (authors) 
for auth in authors:  
	files = os.listdir(path+auth+'/');
	#print (files)
	label_id = len(label_index)
	label_index[auth] = label_id
	for file in files:
		f=open(path+auth+'/'+file, 'r')
		data = f.read().replace('\n', '')
#		print(data)
		#print (path+auth+'/'+file, os.path.exists(path+auth+'/'+file),'size',len(data),auth)   
		texts.append(data)
		labels.append(label_id)
		f.close()
#print("Texts")
#print(texts)
#print("Labels")
#print(labels)


GLOVE_DIR = 'glove.6B/'
MAX_SEQUENCE_LENGTH = 5000
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.1

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#print(word_index)

print('Indexing word vectors.')
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#print(data)
labels = to_categorical(np.asarray(labels))
#print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

X_train = data
Y_train = labels

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word,i in word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector[:EMBEDDING_DIM]

#print(embedding_matrix)
embedding_layer = Embedding(embedding_matrix.shape[0],
				embedding_matrix.shape[1],
				weights=[embedding_matrix],
				input_length=MAX_SEQUENCE_LENGTH,
				trainable=False)


model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(64,5,activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(64,5,activation='relu'))
model.add(MaxPooling1D(5))
#model.add(Dropout(0.30))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
#model.add(Dropout(0.10))
model.add(Dense(16,activation='relu'))
#model.add(Dropout(0.45))
model.add(Dense(len(label_index),activation='softmax'))
#sgd = SGD (lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
tensorboard=TensorBoard(log_dir="logs/{}".format(time()))
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=25,batch_size=10,callbacks=[tensorboard])
scores = model.evaluate(x_val, y_val, verbose=0)
#save the model
print(scores)
model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
