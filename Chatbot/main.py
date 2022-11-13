# Building a Chatbot from Scratch 

# In this project we will build a chatbot from scratch using the corenell University's Movie Dialogue corpus.
# We will be using a deep learning based architecture with the main components as a lstm based encoder and decoder.

# install textBlob
# install pyspellchecker

import keras
import sklearn
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import re
from spellchecker import SpellChecker
