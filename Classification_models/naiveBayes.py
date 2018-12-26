import argparse
import numpy as np
import gensim
import logging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

#For reproducibility of results
np.random.seed(100)

parser = argparse.ArgumentParser(description='Implement a support vector machines method for classification')

parser.add_argument('-t', '--test_size', type=float, required = False,
                    help='fraction of the data used for test (between 0 and 1)')

args = parser.parse_args()

if(args.test_size == None):
    TEST_SIZE = 0.2
else:
    TEST_SIZE = args.test_size

data =  pd.read_csv('../data/data_preprocessed.csv')

data_train, data_test = train_test_split(data, test_size=TEST_SIZE) # random_state = 0
print('Size of test set: ', str(len(data_test)), ' size of train set: ', str(len(data_train)))
data_train = data_train.reset_index(drop = True)
data_test = data_test.reset_index(drop = True)

X_train = data_train['Title paper']
y_train = data_train['Label']

X_test = data_test['Title paper']
y_test = data_test['Label']

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train)

X_test_tfidf = tfidf_vect.transform(X_test)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)

print('Naive Bayes gives accuracy of {:.2f} when applied to concatenated titles'.format(np.mean(clf.predict(X_test_tfidf) == y_test)))
