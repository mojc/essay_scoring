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
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_base_en_uncased",trainable=True)

encoder_inputs = preprocessor(text_input)
