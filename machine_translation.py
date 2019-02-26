# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:22:48 2019

@author: Melvin
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import download
import yat
import os

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Embedding, GRU, Dense
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

a = download.download()
b = download.download1()
eng = open(a, "r").read().splitlines()[:10000]
jap = open(b, "r", encoding="utf-8").read().splitlines()[:10000]
start = 'ssss'
end = 'eeee'
jap = [start + " " + sentence + " " + end for sentence in jap]
#segmenter = tinysegmenter.TinySegmenter()
num_words = 20000

#Create tokenizer for to support both languages

class TokenizeWrapper(Tokenizer):
    def __init__(self, texts, reverse=False, padding=None,num_words=None,
                 truncating=None, japanese=None):
        if japanese:
            self.tokenizer = yat.Tokenizer()
            self.tokenizer.fit_on_texts(texts)
            self.dictionary = {k[0]:v for k,v in self.tokenizer.token2id.items()}
            self.reverse_tokens_jap = dict(zip(self.dictionary.values(), self.dictionary.keys()))
            self.tokens = self.tokenizer.texts_to_sequences(texts)
            
            if reverse:
                self.tokens = [list(reversed(token)) for token in self.tokens]
            
            self.num_tokens = [len(token) for token in self.tokens]
            self.max_num = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
            self.max_num = int(self.max_num)
            self.sequences_padded = pad_sequences(self.tokens, maxlen=self.max_num,
                                                  padding=padding,
                                                  truncating=truncating)

        else:
            Tokenizer.__init__(self, num_words=num_words)
            self.fit_on_texts(texts)
            self.tokens = self.texts_to_sequences(texts)
            self.reverse_tokens = dict(zip(self.word_index.values(),
                                           self.word_index.keys()))
    
            if reverse:
                self.tokens = [list(reversed(token)) for token in self.tokens]
            
            self.num_tokens = [len(token) for token in self.tokens]
            self.max_num = np.mean(self.num_tokens) + 2 * np.std(self.num_tokens)
            self.max_num = int(self.max_num)
            self.sequences_padded = pad_sequences(self.tokens, maxlen=self.max_num,
                                                  padding=padding,
                                                  truncating=truncating)
    
    def sequences_to_text(self, token, japanese=False):
        if japanese:
            sentence = "".join(self.tokenizer.sequences_to_texts([token]))
        else:
            words = [self.reverse_tokens[t] for t in token]
            sentence = "".join(words)
        return sentence
    
    def token_to_word(self, token):
        word = " " if token == 0 else self.reverse_tokens_jap[token]
        return word 
    
    def text_to_sequences(self, text, reverse=False, padding=False):
        tokens = self.texts_to_sequences([text])
        tokens = np.array(tokens)

        if reverse:
            # Reverse the tokens.
            tokens = np.flip(tokens, axis=1)

            # Sequences that are too long should now be truncated
            # at the beginning, which corresponds to the end of
            # the original sequences.
            truncating = 'pre'
        else:
            # Sequences that are too long should be truncated
            # at the end.
            truncating = 'post'

        if padding:
            # Pad and truncate sequences to the given length.
            tokens = pad_sequences(tokens,
                                   maxlen=self.max_num,
                                   padding='pre',
                                   truncating=truncating)

        return tokens
    
englishTokenizer = TokenizeWrapper(eng, reverse=True, padding="pre",
                                   truncating="pre", num_words=num_words)
japaneseTokenizer = TokenizeWrapper(jap, reverse=False, padding="post",
                                    truncating="post", num_words=num_words, japanese=True)
english_padded = englishTokenizer.sequences_padded
japanese_padded = japaneseTokenizer.sequences_padded
token_start = japaneseTokenizer.dictionary[start.strip()]
token_end = japaneseTokenizer.dictionary[end.strip()]

#Create training data

encoder_input_data = english_padded
decoder_input_data = japanese_padded[:, :-1]
decoder_output_data = japanese_padded[:, 1:]

#Create encoder networks
embedding_size = 128
state_size = 512
num_layers = 3
encoder_input = Input(shape=(None, ), name="encoder_input")
encoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size,
                     name="encoder_embedding")
encoder_network_gru = {}
for i in range(1,num_layers):
    encoder_network_gru["encoder_gru{}".format(i)] = GRU(state_size,
                return_sequences=True, name="encoder_gru{}".format(i))
encoder_network_gru["encoder_gru{}".format(num_layers)] = GRU(state_size,
            return_sequences=False, name="encoder_gru{}".format(num_layers))

def connect_encoder_networks(encoder_network_gru):
    net = encoder_input
    net = encoder_embedding(net)
    for k,v in encoder_network_gru.items():
        net = v(net)
    encoder_output = net
    return encoder_output

encoder_output = connect_encoder_networks(encoder_network_gru)

# Create decoder networks

decoder_input = Input(shape=(None, ), name="decoder_input")
decoder_initial_input = Input(shape=(state_size, ),
                              name="decoder_initial_input")
decoder_embedding = Embedding(input_dim=num_words, output_dim=embedding_size,
                              name="decoder_embedding")
decoder_network_gru = {}
for i in range(1,num_layers+1):
    decoder_network_gru["decoder_gru{}".format(i)] = GRU(state_size,
                        return_sequences=True, name="decoder_gru{}".format(i))
decoder_dense = Dense(num_words, activation="linear", name="decoder_output")

def connect_decoder_networks(decoder_network_gru, initial_input):
    net = decoder_input
    net = decoder_embedding(net)
    for k, v in decoder_network_gru.items():
        net = v(net, initial_state=initial_input)
    decoder_output = decoder_dense(net)
    return decoder_output

decoder_output = connect_decoder_networks(decoder_network_gru, encoder_output)
model_train = Model(inputs=[encoder_input, decoder_input],
                    outputs=[decoder_output])
model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])
decoder_output_temp = connect_decoder_networks(
        decoder_network_gru, decoder_initial_input)
model_decoder = Model(inputs=[encoder_input, decoder_input, decoder_initial_input],
                      outputs=[decoder_output])

def loss_function(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype="int32", shape=(None, None))
model_train.compile(optimizer=optimizer, loss=loss_function,
                                     target_tensors=[decoder_target])

cwd = os.getcwd()
path_checkpoint = cwd + "/data/weights-1-{epoch:02d}-{loss:.4f}.hdf5"
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)


callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=3, verbose=1)

callback_tensorboard = TensorBoard(log_dir='./21_logs/',
                                   histogram_freq=0,
                                   write_graph=False)

callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard]

try:
    model_train.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

x_train = {
        "encoder_input" : encoder_input_data,
        "decoder_input" : decoder_input_data
        }

y_train = {
        "decoder_output" : decoder_output_data
        }

validation_split = 10000 / len(encoder_input_data)
model_train.fit(x=x_train, y=y_train, batch_size=512, epochs=1,
                validation_split=validation_split, callbacks=callbacks)
