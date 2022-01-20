import keras.models
import os
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD
MAX_SEQUENCE_LENGTH = 5000
MAX_WORDS = 40000

def init():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	#sgd = SGD (lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	loaded_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
	graph = tf.get_default_graph()
	return loaded_model,graph

global model,graph
model,graph = init()

def predict():
	predict_text = []
	f = open('predictrud.txt','r')
	data1 = f.read().replace('\n','')
	predict_text.append(data1)
	f.close
	label_index={}
	authors = sorted(os.listdir('Dataset/'))
	for auth in authors:
		label_id = len(label_index)
		label_index[auth] = label_id
	tokenizer = Tokenizer(num_words = MAX_WORDS)
	tokenizer.fit_on_texts(predict_text)
	word_index = tokenizer.word_index
	sequences = tokenizer.texts_to_sequences(predict_text)
	data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH)
	X_data = data
	with graph.as_default():
		out = model.predict(X_data)
		print(out)
		print(np.argmax(out,axis=1))
		print(label_index)
		#print(out[label_index])
if __name__ == "__main__":
	predict()
