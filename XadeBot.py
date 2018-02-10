'''
## Add file to git
git add FILE_TO_ADD
## Commit change with descriptive message
git commit -m "DESCRIPTION FOR COMMIT HERE!"
## Push commit to master branch
git push origin master
'''
import re
import tensorflow as tf
import numpy as np
import time

questions = open('questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
answers = open('answers.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

## Clean text of unwanted character combinations
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"at's", "at is", text)
    text = re.sub(r"re's", "re is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"\'", " would", text)
    text = re.sub(r"newlinechar", "", text)
    text = re.sub(r"\&amp", "", text)
    text = re.sub(r"[-()\"^#/@;:<>{}*+=~|.!?,]", "", text)
    return text

## Clean up the questions file and store results into list
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
## Clean up the answers file and store results into list
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

## Map each word to its number of occurences and store results into dictionary
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

## Creates two dictionaries that map the questions words and the answers words to a unique integer
## This makes it easier to tokenize words
threshold = 20
questions_words_2_int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questions_words_2_int[word] = word_number
        word_number += 1
answers_words_2_int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answers_words_2_int[word] = word_number
        word_number += 1
        
## Add last tokens to these two dictionaries
tokens = ['<PAD>','<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questions_words_2_int[token] = len(questions_words_2_int) + 1
for token in tokens:
    answers_words_2_int[token] = len(answers_words_2_int) + 1

## Creating the inverse dictionary of the answers_words_2_int dictionary
answers_ints_2_word = {w_i: w for w, w_i in answers_words_2_int.items()}

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_2_int:
            ints.append(questions_words_2_int['<OUT>'])
        else:
            ints.append(questions_words_2_int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answers_words_2_int:
            ints.append(answers_words_2_int['<OUT>'])
        else:
            ints.append(answers_words_2_int[word])
    answers_into_int.append(ints)

## Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
## Creates Placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

## Preprocessing the targets
def preprocess_targets(targets, word_2_int, batch_size):
    left_side = tf.fill([batch_size, 1], word_2_int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

## Creates the Encoder RNN Layer
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

            
            
            
            
            
            
            