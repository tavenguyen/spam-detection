import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')

global_vocab = set()
spam_dict = {}
total_words_in_spam = 0

##################################################################
file_obj = open("SMSSpamCollection.txt", mode = 'r', encoding = 'utf-8')
while (1):
    text = file_obj.readline()
    if(not text):
        break
    
    parts = text.strip().split(maxsplit=1)
    label = parts[0]
    message = parts[1]
    
    words = nltk.tokenize.word_tokenize(message)

    for w in words:
        global_vocab.add(w)

    if(label == 'spam'):
        for w in words:
            total_words_in_spam += 1
            if w not in spam_dict:
                spam_dict[w] = 0
            spam_dict[w] += 1

file_obj.close()

##################################################################
vocab_size = len(global_vocab)

# P(Words | Spam) = (Count(W_i in Spam) + 1) / (Total Words In Spam + V)
word = 'free'
probability = (spam_dict.get(word, 0) + 1) / (total_words_in_spam + vocab_size)
print('Probability of \'', word, '\'=', probability)