import numpy as np
import pandas as pd

#Preparing Data
read = pd.read_csv('email.csv', sep=',')
data = read.drop(read.columns[0], axis=1)
train = data.head(int(len(data)*0.8))
test_temp = data.drop(train.index)
test = test_temp.drop(test_temp.columns[-1], axis=1)
spam_temp = train[train['Prediction'] == 1]
spam = spam_temp.drop(spam_temp.columns[-1], axis=1)
nonspam_temp = train[train['Prediction'] == 0]
nonspam = nonspam_temp.drop(nonspam_temp.columns[-1], axis=1)

#Training
n_words = len(train.columns) - 1  #vocabulary
n_spam = spam.apply(len).sum()
n_nonspam = nonspam.apply(len).sum()
p_spam = len(spam) / len(train)  #prior
p_nonspam = len(nonspam) / len(train)  #prior

n_word_spam = spam.sum()
n_word_nonspam = nonspam.sum()

p_word_spam = (n_word_spam+1) / (n_spam+n_words)  #likelihood
p_word_nonspam = (n_word_nonspam+1) / (n_nonspam+n_words)  #likelihood

#Test
correct = 0
product_word_spam = 0
product_word_nonspam = 0
p_spam_word = [1] * len(test)
p_nonspam_word = [1] * len(test)
for row in range(0, len(test)):
    for column in range(0, len(test.columns)):
        product_word_spam += pow(np.log(p_word_spam[column]), test.iloc[row, column])
        product_word_nonspam += pow(np.log(p_word_nonspam[column]), test.iloc[row, column])
    p_spam_word[row] = np.log(p_spam) + product_word_spam
    p_nonspam_word[row] = np.log(p_nonspam) + product_word_nonspam
    if p_spam_word[row] > p_nonspam_word[row] and test_temp.iloc[row][-1] == 1:
        correct += 1
    if p_spam_word[row] < p_nonspam_word[row] and test_temp.iloc[row][-1] == 0:
        correct += 1
accuracy = (correct / len(test)) * 100
print(correct)
print(accuracy)




