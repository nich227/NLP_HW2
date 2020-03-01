'''
Name: Kevin Chen
NetID: nkc160130
CS 6320
Due: 3/2/2020
Dr. Moldovan
Version: Python 3.8.0
'''

import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import sys
import os

# Check Python version
if sys.version_info[0] < 3:
    raise Exception("ERROR: This program requires Python 3.x to run.")
if sys.version_info[1] < 6 or (sys.version_info[1] == 6 and sys.version_info[2] < 1):
    print("WARNING: This program will run best on Python version >=3.6.1.")

# Invalid number of arguments provided
if len(sys.argv)-1 != 6:
    print("ERROR: Invalid number of arguments!")
    print("Usage: hw2.py <training-set> <test-set> <representation> <classifier> <stop-words> <regularization>")
    exit(1)

# Get all arguments and put into variables (for readability)
train_loc = sys.argv[1]
test_loc = sys.argv[2]
represent = sys.argv[3]
classify = sys.argv[4]
stop_wd = int(sys.argv[5])
reg = sys.argv[6]

# Validating arguments
if os.path.exists(train_loc) == False:
    print("ERROR: " + train_loc + ": The directory does not exist.")
    exit(1)
if os.path.exists(test_loc) == False:
    print("ERROR: " + test_loc + ": The directory does not exist.")
    exit(1)
if represent != 'bow' and represent != 'tfidf':
    print('ERROR: Invalid representation value ' + represent + '.')
    exit(1)
if classify != 'nbayes' and classify != 'regression':
    print('ERROR: Invalid classifier ' + classify + '.')
    exit(1)
if stop_wd != 0 and stop_wd != 1:
    print('ERROR: Invalid stop word value ' + stop_wd + '.')
    exit(1)
if reg != 'no' and reg != 'l1' and reg != 'l2':
    print('ERROR: Invalid regularization ' + reg + '.')


# Opens a file and extracts text, adds to corpora


def openCorpora(fileName):
    sentences = ""
    try:
        with open(fileName, 'r', encoding='utf8') as input:
            for ln in input:
                for wd in ln.split():
                    wd = wd.lower()
                    puncts = [
                        '.', '!', '?', '-', '(', ')', '\'', '{', '}', ',', '*', '[', ']', '\"', '/', '\\', '>', '<br', ':', '\t']
                    # Getting rid of punctuation characters
                    for punct in puncts:
                        if punct == '-' or punct == '/' or punct == '\t':
                            wd = wd.replace(punct, ' ')
                        else:
                            wd = wd.replace(punct, '')

                    # Add word to the sentences in file
                    sentences += (wd + ' ')

    except FileNotFoundError:
        print("ERROR:", fileName, "not found!")
        exit(1)

    return sentences[:-1]

# Import positive and negative documents and convert to a dataframe


def getDf(loc):
    posCorpora = []
    negCorpora = []

    # Collecting positive corpora
    for fileName in os.listdir(loc+'/pos'):
        if fileName.endswith('.txt'):
            posCorpora.append(openCorpora(loc+'/pos/'+fileName))

    # Collecting negative corpora
    for fileName in os.listdir(loc+'/neg'):
        if fileName.endswith('.txt'):
            negCorpora.append(openCorpora(loc+'/neg/'+fileName))

    # Converting lists to dataframes
    label = []
    for index in posCorpora:
        label.append(1)
    posDf = pd.DataFrame(list(zip(label, posCorpora)),
                         columns=['label', 'document'])

    label = []
    for index in negCorpora:
        label.append(0)
    negDf = pd.DataFrame(list(zip(label, negCorpora)),
                         columns=['label', 'document'])

    # Combining dataframes
    return pd.concat([posDf, negDf], ignore_index=True)


# Driver of the program
orig_wd = os.getcwd()
start_time = time.time()

# Retrieve and process training data
trainDf = getDf(train_loc)
testDf = getDf(test_loc)

# Getting x and y train and test (for readability only)
X_train = trainDf['document']
y_train = trainDf['label']
X_test = testDf['document']
y_test = testDf['label']

# Stop word list - from https://www.ranks.nl/stopwords
stop_wds_list = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off',
                 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'were', 'werent', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves']

# Initialize CountVectorizer
if represent == 'bow':
    # Stop words
    if stop_wd == 0:
        vec = CountVectorizer()
    if stop_wd == 1:
        vec = CountVectorizer(stop_words=stop_wds_list)

# Initialize TfidfVectorizer
if represent == 'tfidf':
    if stop_wd == 0:
        vec = TfidfVectorizer()
    if stop_wd == 1:
        vec = TfidfVectorizer(stop_words=stop_wds_list)

# Fit training data
train_data = vec.fit_transform(X_train)

# Transform test data
test_data = vec.transform(X_test)

# Perform Naive Bayes classification
if classify == 'nbayes':
    nbayes = MultinomialNB()
    nbayes.fit(train_data, y_train)
    predictions = nbayes.predict(test_data)

# Perform Logistic Regression classification
if classify == 'regression':
    # No penalty
    if reg == 'no':
        log_reg = LogisticRegression(solver='saga', penalty='none')
    else:
        log_reg = LogisticRegression(solver='saga', penalty=reg)
    log_reg.fit(train_data, y_train)
    predictions = log_reg.predict(test_data)

# Report performance
print('---------------')
print('| Performance |')
print('---------------')
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions))
print('Recall score: ', recall_score(y_test, predictions))
print('F1 score: ', f1_score(y_test, predictions))


# End of program
print('-----\n', 'HW2 took', round(time.time() -
                                    start_time, 4), 'seconds to complete.')
