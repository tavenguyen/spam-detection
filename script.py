import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')

global_vocab = dict()
spam_dict = {}
total_samples = 0
total_spam_mails = 0
total_words = 0
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
        global_vocab[w] = global_vocab.get(w, 0) + 1
        total_words += 1

    if(label == 'spam'):
        total_spam_mails += 1
        for w in words:
            total_words_in_spam += 1
            if w not in spam_dict:
                spam_dict[w] = 0
            spam_dict[w] += 1
    total_samples += 1

file_obj.close()

##################################################################
vocab_size = len(global_vocab)

# P(Word | Spam) = (Count(W_i in Spam) + 1) / (Total Words In Spam + V)
# P(Spam | Word) = P(Word | Spam) . P(Spam) / P(Word)
def calculateProbabilityOfWord(word):
    probability_spam = total_spam_mails / total_samples
    probability_word = global_vocab.get(word, 0) / total_words
    probability_word_spam = (spam_dict.get(word, 0) + 1) / (total_words_in_spam + vocab_size)
    return probability_word_spam * probability_spam / probability_word

sentence = "Free money for your live!"
words = nltk.tokenize.word_tokenize(sentence)
probability = 1.0
for word in words:
    probability *= calculateProbabilityOfWord(word)

print("Probability of", sentence, "to be 'spam email' is:", probability) 
