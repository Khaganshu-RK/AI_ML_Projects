# Building a Chatbot from Scratch 

# In this project we will build a chatbot from scratch using the corenell University's Movie Dialogue corpus.
# We will be using a deep learning based architecture with the main components as a lstm based encoder and decoder.


#Importing Libraries
import re
import nltk
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from spellchecker import SpellChecker


#Download the glove model available at https://nlp.stanford.edu/projects/glove/
#Specification : Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip


#Hyperparameter tuning 
RAND_STATE=np.random.seed(42)
BATCH_SIZE = 32
NUM_EPOCHS = 10
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 10000
DATA_PATH = 'Chatbot/Data/movie_lines_cleaned.txt'
WEIGHT_FILE_PATH = 'Chatbot/Data/model-weights.h5'
GLOVE_MODEL = "Chatbot/Data/glove.twitter.27B/glove.twitter.27B." + str(GLOVE_EMBEDDING_SIZE) + "d.txt"
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
ignore_words = ['?', '!']


# Function **in_white_list()** checks whether the words in **movie_lines_cleaned.txt** formed with Valid characters.

#defines the valid characters for the chatbot
def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False



# Load the glove word embedding in to a dictionary where the key is a unique word token and the value is a d dimension vector

def load_glove_vector():
    _word2embedding = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        #print(words)
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2embedding[word] = embeds
        
    file.close()
    print('Glove Loaded !')
    file.close()
    return _word2embedding