# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:45:30 2019

@author: admin
"""

import machine_translation
import numpy as np

def translator(text, true_output_text=None):
    input_tokens = machine_translation.englishTokenizer.text_to_sequences(
            text,reverse=True, padding=True)
    print(input_tokens)
    initial_state = machine_translation.model_encoder.predict(input_tokens)

    # Max number of tokens / words in the output sequence.
    max_tokens = machine_translation.japaneseTokenizer.max_num

    shape = (1, max_tokens)
    machine_translation.decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    # The first input-token is the special start-token for 'ssss '.
    token_int = machine_translation.token_start

    # Initialize an empty output-text.
    output_text = ''

    # Initialize the number of tokens we have processed.
    count_tokens = 0
    
    encoder_input_data_temp = machine_translation.pad_sequences(input_tokens, machine_translation.englishTokenizer.max_num,padding="pre", truncating="pre")

    # While we haven't sampled the special end-token for ' eeee'
    # and we haven't processed the max number of tokens.
    while token_int != machine_translation.token_end and count_tokens < max_tokens:
        # Update the input-sequence to the decoder
        # with the last token that was sampled.
        # In the first iteration this will set the
        # first element to the start-token.
        machine_translation.decoder_input_data[0, count_tokens] = token_int

        # Wrap the input-data in a dict for clarity and safety,
        # so we are sure we input the data in the right order.
        x_data = \
        {
            'decoder_initial_input': initial_state,
            'decoder_input': machine_translation.decoder_input_data,
            'encoder_input': encoder_input_data_temp
        }


        # Input this data to the decoder and get the predicted output.
        machine_translation.decoder_output = machine_translation.model_decoder.predict(x_data)

        # Get the last predicted token as a one-hot encoded array.
        token_onehot = machine_translation.decoder_output[0, count_tokens, :]
        
        # Convert to an integer-token.
        token_int = np.argmax(token_onehot)

        # Lookup the word corresponding to this integer-token.
        sampled_word = machine_translation.japaneseTokenizer.token_to_word(
                token_int)

        # Append the word to the output-text.
        output_text += " " + sampled_word

        # Increment the token-counter.
        count_tokens += 1

    # Sequence of tokens output by the decoder.
    output_tokens = machine_translation.decoder_input_data[0]
    
    # Print the input-text.
    print("Input text:")
    print(text)
    print()

    # Print the translated output-text.
    print("Translated text:")
    print(output_text)
    print()

    # Optionally print the true translated text.
    if true_output_text is not None:
        print("True output text:")
        true_output = "".join(true_output_text.split()[1:-1])
        print(true_output)
        print()
        
idx = 3
translator(text=machine_translation.eng[idx],
          true_output_text=machine_translation.jap[idx])