# Building a Chatbot from Scratch 

# In this project we will build a chatbot from scratch using the corenell University's Movie Dialogue corpus.
# We will be using a deep learning based architecture with the main components as a lstm based encoder and decoder.


#Importing Libraries
import re
import os
import nltk
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Embedding, LSTM
from keras_preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from spellchecker import SpellChecker
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
DATA_SET_NAME = 'Chatbot/Data'
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

word2embedding = load_glove_vector()



# Data Preparation
## Basic Text Preprocessing: This basic pre-processing is necessary because if the GLOVE Model did'nt understand the word ,then it will not create the Embedding for the word.
  ## Replace Contractions
  ## whitelist_lines()
  ## Spelling Correction Library
     ## TextBlob
     ## pyspellchecker

nltk.download('punkt')

target_counter = Counter()
lines = open(DATA_PATH, 'rt', encoding='utf8').read().split('\n')
input_texts = []
target_texts = []
prev_words = []
contraction_dict = {
  "aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have","couldn't": "could not","couldn't've": "could not have",
  "didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have","hasn't": "has not","haven't": "have not","he'd": "he would",
  "he'd've": "he would have","he'll": "he will","he'll've": "he will have","he's": "he is","how'd": "how did","how'd'y": "how do you","how'll": "how will","how's": "how is","I'd": "I would",
  "I'd've": "I would have","I'll": "I will","I'll've": "I will have","I'm": "I am","I've": "I have","isn't": "is not","it'd": "it had",
  "it'd've": "it would have","it'll": "it will","it'll've": "it will have","it's": "it is","let's": "let us","ma'am": "madam",
  "mayn't": "may not","might've": "might have","mightn't": "might not","mightn't've": "might not have","must've": "must have","mustn't": "must not","mustn't've": "must not have","needn't": "need not",
  "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not","oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
  "shan't've": "shall not have","she'd": "she would","she'd've": "she would have","she'll": "she will","she'll've": "she will have","she's": "she is","should've": "should have",
  "shouldn't": "should not","shouldn't've": "should not have","so've": "so have","so's": "so is",
  "that'd": "that would","that'd've": "that would have","that's": "that is","there'd": "there had","there'd've": "there would have","there's": "there is","they'd": "they would","they'd've": "they would have",
  "they'll": "they will","they'll've": "they will have","they're": "they are","they've": "they have","to've": "to have",
  "wasn't": "was not","we'd": "we had","we'd've": "we would have","we'll": "we will","we'll've": "we will have","we're": "we are","we've": "we have",
  "weren't": "were not","what'll": "what will","what'll've": "what will have","what're": "what are",
  "what's": "what is","what've": "what have","when's": "when is","when've": "when have","where'd": "where did","where's": "where is","where've": "where have","who'll": "who will",
  "who'll've": "who will have","who's": "who is","who've": "who have","why's": "why is","why've": "why have",
  "will've": "will have","won't": "will not","won't've": "will not have","would've": "would have",
  "wouldn't": "would not","wouldn't've": "would not have","y'all": "you all","y'alls": "you alls","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
  "you'd": "you had","you'd've": "you would have","you'll": "you you will","you'll've": "you you will have","you're": "you are","you've": "you have"
  }


#Replace Contractions
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)

lines = [replace_contractions(line) for line in lines]



# whitelist_lines function tries to remove :
 ## Removed HTML Tags
 ## Remove - and replace with ' ' (space)
 ## Allowed Chacters -> a-z A-Z ! space ,? .
 ## Repeating Punctuations (?.!)[Ex - Hi !!! to Hi !]
 ## Repeating Characters in the word [Ex - wayyy to way]
 ## Repeating Characters in between the words
 ## Period (.) before word and after word [Ex - .lion to lion]
 ## Double space,newline,tab to single space

def whitelist_lines(line):
  line = re.sub('(<.*?>)','',line)
  line = re.sub(r'-+',r' ',line)
  line = re.sub(r'[^(a-zA-Z!\s,?\.)]*',r'',line)
  line = re.sub(r'([!?.,]){1,}\1',r'\1',line)
  line = re.sub(r'\b(\S*?)(.)\2{2,}\b',r'\1\2',line)
  line = re.sub(r'\b(\S*?)(.)\2{2,}(\S+)\b',r'\1\2\2\3',line)
  line = re.sub(r'([.])',r' . ',line)
  line = re.sub(r'\s+',r' ',line)
  return line.strip()

lines = [whitelist_lines(line) for line in lines]


#spelling_correction_textBlob functions tries to correct the mis-spelled word in the lines.
def spelling_correction_textBlob(text):
  text = TextBlob(text).correct()
  return str(text)

for line in lines:
    next_words = [w.lower() for w in nltk.word_tokenize(line)]
    if len(next_words) > MAX_TARGET_SEQ_LENGTH:
        next_words = next_words[0:MAX_TARGET_SEQ_LENGTH]
    if len(prev_words) > 0:
        input_texts.append(prev_words)
        target_words = next_words[:]
        target_words.insert(0, 'start')
        target_words.append('end')
        for w in target_words:
            target_counter[w] += 1
        target_texts.append(target_words)
    prev_words = next_words

print('Length of Input texts :',len(input_texts))
print('Length of Target texts :',len(target_texts))


#Let's see some of the training examples
for idx, (input_words, target_words) in enumerate(zip(input_texts, target_texts)):
    if idx > 10:
        break
    print([input_words, target_words])


target_word2idx = dict()
'''create a target word to id dictionary called target_word2idx.
2 to 3 lines '''
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_word2idx[word[0]] = idx + 1

if 'unk' not in target_word2idx:
    target_word2idx['unk'] = 0

'''create a target to id dictionary called target_idx2word . Approx ~1 line'''

target_idx2word = dict([(idx, word) for word, idx in target_word2idx.items()])

num_decoder_tokens = len(target_idx2word)

np.save( DATA_SET_NAME + '/word-glove-target-word2idx.npy', target_word2idx)
np.save( DATA_SET_NAME + '/word-glove-target-idx2word.npy', target_idx2word)
print(len (target_word2idx.keys())==len (target_idx2word.keys())==MAX_VOCAB_SIZE+1)



target_word2idx = np.load(DATA_SET_NAME + '/word-glove-target-word2idx.npy',allow_pickle=True).item()
target_idx2word = np.load( DATA_SET_NAME + '/word-glove-target-idx2word.npy',allow_pickle=True).item()



input_texts_word2em = []
encoder_max_seq_length = 0
decoder_max_seq_length = 0
corrected_noembwords = []
noembwords = []
unableToEmbed_spellchecker = []
obj = SpellChecker()

for input_words, target_words in zip(input_texts, target_texts):
    encoder_input_wids = []
    for w in input_words:
        embeddings=np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if w in word2embedding:
          embeddings = word2embedding[w]
          encoder_input_wids.append(embeddings)
        else:
          noembwords.append(w)
          w1 = spelling_correction_textBlob(w)
          if w1 in word2embedding:
            embeddings = word2embedding[w1]
            encoder_input_wids.append(embeddings)
          else:
            corrected_noembwords.append(w)
            w2 = obj.correction(w)
            if w2 in word2embedding:
              embeddings = word2embedding[w2]
              encoder_input_wids.append(embeddings)
            else:
              unableToEmbed_spellchecker.append(w)
              encoder_input_wids.append(embeddings)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print('Context : ',context)
np.save( DATA_SET_NAME + '/word-glove-context.npy', context)



context = np.load(DATA_SET_NAME + '/word-glove-context.npy',allow_pickle=True).item()

encoder_max_seq_length = context['encoder_max_seq_length']
decoder_max_seq_length = context['decoder_max_seq_length']
num_decoder_tokens = context['num_decoder_tokens']

def generate_batch(input_word2em_data, output_text_data):
    num_batches = len(input_word2em_data) // BATCH_SIZE
    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * BATCH_SIZE
            end = (batchIdx + 1) * BATCH_SIZE
            '''Fill your code here. 5 to 10 lines'''
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end], encoder_max_seq_length)
            decoder_target_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
                for idx, w in enumerate(target_words):
                    w2idx = target_word2idx['unknown']  # default unknown
                    if w in target_word2idx:
                        w2idx = target_word2idx[w]
                    if w in word2embedding:
                        decoder_input_data_batch[lineIdx, idx, :] = word2embedding[w]
                    if idx > 0:
                        decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, target_texts, test_size=0.2, random_state=42)
train_gen = generate_batch(Xtrain, Ytrain)

#Encoder layers,inputs,outputs
encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE))
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True)
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

#Decoder layers - input,output,LSTM,Dense
decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE))
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True)
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                 initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
#model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])

#
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
#


import json
# serialize model to JSON
model_json = model.to_json()
with open(DATA_SET_NAME+'/word-architecture.json', "w") as json_file:
    json_file.write(model_json)

model.summary()

# Just Checking the size of the Training and testing set.
print('Length Xtrain :',len(Xtrain))
print('Length Xtest :',len(Xtest))
print('Length Ytrain :',len(Ytrain))
print('Length Ytest :',len(Ytest))

test_gen = generate_batch(Xtest, Ytest)
train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)

#fitting of chatbot model
history = model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS,
                    verbose=1, validation_data=test_gen, validation_steps=test_num_batches, callbacks=[checkpoint])


model.save_weights(WEIGHT_FILE_PATH)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()




from keras.models import model_from_json
filename = DATA_SET_NAME+'/Industry Grade Project - Building a Chatbot/model-weights.h5'

json_file = open(DATA_SET_NAME+'/Industry Grade Project - Building a Chatbot/word-architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(filename)
print('Model Loaded !')


loaded_model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop')

encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

def reply(input_text):
        input_seq = []
        input_emb = []
        for word in nltk.word_tokenize(input_text.lower()):
            if not in_white_list(word):
                continue
            emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
            if word in word2embedding:
                emb = word2embedding[word]
            
            input_emb.append(emb)
            #print('input_emb --',input_emb)
        input_seq.append(input_emb)
        #print('input_seq --',input_seq)
        input_seq = pad_sequences(input_seq,encoder_max_seq_length)
        states_value = encoder_model.predict(input_seq)
        #print('States Value --',states_value)
        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = word2embedding['start']
        #print('Target Sequence - ',target_seq)
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            #print('Output tokens -',output_tokens,'h -',h,'c -',c)
            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'start' and sample_word != 'end':
                target_text += ' ' + sample_word

            if sample_word == 'end' or target_text_len >= decoder_max_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
            if sample_word in word2embedding:
                target_seq[0, 0, :] = word2embedding[sample_word]

            states_value = [h, c]
        return target_text.strip()

def test_model(ques):
    rikisays=reply(ques)
    return rikisays

mayank= 'Are you free tommorow?'
print('Mayank :',mayank)
riki = test_model(mayank)
print('Riki :',riki)   
