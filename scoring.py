
#import keras_nlp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

import keras
import tensorflow as tf
import numpy as np
import pandas as pd

# data = keras.utils.text_dataset_from_directory('learning-agency-lab-automated-essay-scoring-2/')
data = pd.read_csv("learning-agency-lab-automated-essay-scoring-2/train.csv")
essays = data.full_text.to_list()
labels = data.score.to_list() # maybe one hot and cross entropy

num_words=10000 # words that occur at least twice
training_size = 14000 # 80% of essays
padding_type='post'
trunc_type='post'

essays = [es[:2500] for es in essays] # cut the long essays

training_sentences = essays[0:training_size]
testing_sentences = essays[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=num_words, oov_token='<oov>')

tokenizer.fit_on_texts(training_sentences)
# sequences = tokenizer.texts_to_sequences(essays)
max_length = max([len(es) for es in essays])

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, 6),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64),
    layers.Dense(1, activation='relu')]
)
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
model.summary()
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), verbose=2, callbacks=callback)

preds = np.round(model.predict(testing_padded))
print(np.mean(abs(preds - testing_labels)))
print(np.mean((np.round(preds, 0) - testing_labels)**2))

# 1.0258
# 1.7741