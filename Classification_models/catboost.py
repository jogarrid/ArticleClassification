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

data =  pd.read_csv('~jose/JoseNinaAlexML4/JoseNinaAlexML/data/data_preprocessed.csv')

### all vectors are now represented as strings. Represent as arrays of floats.
data['Title w2v'] = data['Title w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2v'] = data['Source w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Title w2vtf'] = data['Title w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2vtf'] = data['Source w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])

data_train, data_test = train_test_split(data, test_size=TEST_SIZE) # random_state = 0
print('Size of test set: ', str(len(data_test)), ' size of train set: ', str(len(data_train)))
data_train = data_train.reset_index(drop = True)
data_test = data_test.reset_index(drop = True)
############################
X = np.array([data_train['Title w2v'][i]+[data_train['Num Fired'][i]]+[data_train['Num not fired'][i]] +[data_train['No titles'][i]] for i in range(len(data_train))])
y = data_train['Label'].values

Xtest = np.array([data_test['Title w2v'][i]+[data_test['Num Fired'][i]]+[data_test['Num not fired'][i]] +[data_test['No titles'][i]] for i in range(len(data_test))])
ytest = data_test['Label'].values

############################
