import random
import numpy as np
#import tensorflow as tf
import keras
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Activation
from keras.optimizers.legacy import RMSprop

filepath = keras.utils.get_file('shakespere.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()


#training range
text = text[300000:800000]

#all distinct characters in text with set() function -> Menge
characters = sorted(set(text))

#assignes numbers to all possible characters
char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))


#Predict next characters
SEQ_LENGTH = 40
STEP_SIZE = 3

sentences = []
next_characters = []

#training generation
"""
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_characters.append(text[i+SEQ_LENGTH])

#convert to numeric
#input data
#One dimention for all possible sentences, one for all posistions in the sentences, one for all possible characters
#-> In a specific Sentence at a specific pos if a certain char occurs, set to True/1, all other val remain 0
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool_)

#output data/target data
y = np.zeros((len(sentences), len(characters)), dtype=np.bool_)

#Loop over all sentences just created, enumerate every character in sentence
#set values in input array x
#set taget data in y array
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1
"""


#model training
"""
model = Sequential()
#LSTM = long short term memory: 'remembers' last importent characters
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
#softmax: scales output so sum is always 1.0
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(r=0.01))

model.fit(x, y, batch_size=256, epochs=4)

model.save('shakesperian_text_generator.model')
"""
    

model = keras.models.load_model('shakesperian_text_generator.model')
    
def sample(preds, temperature=1.0):
    """
    -takes prediction of model choses one of the suggestions from the model
    -high temp: more experimental
    -low temp: conservative
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum (exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(length, temperature):
    start_index = random.randint(0, len(text)-SEQ_LENGTH - 1)
    generated = ''
    #starting 'base' text
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    
    return generated

print(generate_text(800, 0.2))
