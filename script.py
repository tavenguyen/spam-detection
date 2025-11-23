import nltk
import math
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')

global_vocab = dict()
spam_dict = {}
ham_dict = {}
total_samples = 0
total_spam_mails = 0
total_words = 0
total_words_in_spam = 0
total_words_in_ham = 0

##################################################################
file_obj = open("SMSSpamCollection.txt", mode = 'r', encoding = 'utf-8')
while (1):
    text = file_obj.readline()
    if(not text):
        break
    
    parts = text.strip().split(maxsplit=1)
    if(len(parts) < 2):
        continue

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
            spam_dict[w] = spam_dict.get(w, 0) + 1
    else:
        for w in words:
            total_words_in_ham += 1
            ham_dict[w] = ham_dict.get(w, 0) + 1
    
    total_samples += 1

file_obj.close()

##################################################################
vocab_size = len(global_vocab)

# P(Word | Ham), P(Word | Spam)
def calculateProbability(word, type : bool):
    # Spam
    if type == False:
        return (spam_dict.get(word, 0) + 1) / (total_words_in_spam + vocab_size)
    else:
        return (ham_dict.get(word, 0) + 1) / (total_words_in_ham + vocab_size)

sentence = "Miễn phí cho tuần đầu tiên!"
words = nltk.tokenize.word_tokenize(sentence)

# P(Spam)
probability_spam = total_spam_mails / total_samples

# P(Ham)
probability_ham = 1.0 - probability_spam

# Score_{spam}
score_spam = probability_spam
for word in words:
    score_spam  += math.log(calculateProbability(word, False))

# Score_{ham}
score_ham = probability_ham
for word in words:
    score_ham += math.log(calculateProbability(word, True))

print('Score_{ham}:', score_ham, 'Score_{spam}:',score_spam)
if score_ham > score_spam:
    print(sentence, ": ham")
else:
    print(sentence, ": spam")


