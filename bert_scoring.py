import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import  Dense, Dropout
import keras_nlp

import tensorflow_text as text
import tensorflow_hub as hub

from sklearn.model_selection import train_test_split

data = pd.read_csv("learning-agency-lab-automated-essay-scoring-2/train.csv")

X_train, X_val, y_train, y_val = train_test_split(data['full_text'], data['score'], test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape )

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en", trainable=True)
encoder_inputs = preprocessor(text_input)

encoder = keras_nlp.models.BertBackbone.from_preset("bert_base_en", load_weights=True, trainable=False)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]      # [batch_size, 768].
sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

l = tf.keras.layers.Dense(16, activation='relu', name='h1')(pooled_output)
l = tf.keras.layers.Dense(64, activation='relu',name='h2')(l)
l = tf.keras.layers.Dense(32, activation='relu', name='h3')(l)
l = tf.keras.layers.Dense(1,activation='relu',name='output')(l)

model=tf.keras.Model(inputs=[text_input],outputs=[l])
model.compile(optimizer='adam',
              loss='MSE',
              metrics=['accuracy'])
model.summary()
# Non-trainable params: 0 (0.00 B)?

history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

model.evaluate(X_val, y_val)
