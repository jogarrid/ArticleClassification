import argparse
import numpy as np
import gensim
import logging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from catboost import CatBoostClassifier

#For reproducibility of results
np.random.seed(100)

parser = argparse.ArgumentParser(description='Implement Gradient Boosting method for classification')

parser.add_argument('-t', '--test_size', type=float, required = False,
                    help='fraction of the data used for test (between 0 and 1)')
parser.add_argument('-i', '--include_param', type=str, required = False,
                    help='If set to False (type False), do training and testing using only text data. If set to True, include the other discussed parameters')

args = parser.parse_args()

if(args.test_size == None):
    TEST_SIZE = 0.2
else:
    TEST_SIZE = args.test_size

print(args.include_param)
if(args.include_param == None):
    INCLUDE_PARAM = True
else:
    INCLUDE_PARAM = args.include_param == 'True'

data =  pd.read_csv('../data/data_preprocessed.csv')

### all vectors are now represented as strings. Represent as arrays of floats.
data['Title w2v'] = data['Title w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2v'] = data['Source w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Title w2vtf'] = data['Title w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2vtf'] = data['Source w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['sent2vec'] = data['sent2vec'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])

data_train, data_test = train_test_split(data, test_size=TEST_SIZE) # random_state = 0
print('Size of test set: ', str(len(data_test)), ' size of train set: ', str(len(data_train)))
data_train = data_train.reset_index(drop = True)
data_test = data_test.reset_index(drop = True)

if(INCLUDE_PARAM):
    X_train = np.array([data_train['Title w2v'][i]+[data_train['Num Fired'][i]]+[data_train['Num not fired'][i]] +[data_train['No titles'][i]] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['Title w2v'][i]+[data_test['Num Fired'][i]]+[data_test['Num not fired'][i]] +[data_test['No titles'][i]] for i in range(len(data_test))])
    y_test = data_test['Label']
    print('ho')
else: 
    print('hey')
    X_train = np.array([data_train['Title w2v'][i] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['Title w2v'][i] for i in range(len(data_test))])
    y_test = data_test['Label']

model = CatBoostClassifier(iterations=20, learning_rate=1e-1, depth=6, loss_function='Logloss')
model.fit(X_train, y_train)
preds_class = model.predict(X_test)

w2v_res = np.mean(preds_class == y_test)

print('Gradient boosting via catboost gives accuracy of {:.2f} when applied to w2v mean concatenated titles'.format(w2v_res))

if(INCLUDE_PARAM):
    X_train = np.array([data_train['Title w2vtf'][i]+[data_train['Num Fired'][i]]+[data_train['Num not fired'][i]] +[data_train['No titles'][i]] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['Title w2vtf'][i]+[data_test['Num Fired'][i]]+[data_test['Num not fired'][i]] +[data_test['No titles'][i]] for i in range(len(data_test))])
    y_test = data_test['Label']

else: 
    X_train = np.array([data_train['Title w2vtf'][i] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['Title w2vtf'][i] for i in range(len(data_test))])
    y_test = data_test['Label']

model = CatBoostClassifier(iterations=20, learning_rate=1e-1, depth=6, loss_function='Logloss')
model.fit(X_train, y_train)
preds_class = model.predict(X_test)

w2vtf_res = np.mean(preds_class == y_test)

print('Gradient boosting via catboost gives accuracy of {:.2f} when applied to w2v concatenated titles with term frequency'.format(w2vtf_res))

if(INCLUDE_PARAM):
    X_train = np.array([data_train['sent2vec'][i]+[data_train['Num Fired'][i]]+[data_train['Num not fired'][i]] +[data_train['No titles'][i]] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['sent2vec'][i]+[data_test['Num Fired'][i]]+[data_test['Num not fired'][i]] +[data_test['No titles'][i]] for i in range(len(data_test))])
    y_test = data_test['Label']

else: 
    X_train = np.array([data_train['sent2vec'][i] for i in range(len(data_train))])
    y_train = data_train['Label']

    X_test = np.array([data_test['sent2vec'][i] for i in range(len(data_test))])
    y_test = data_test['Label']

model = CatBoostClassifier(iterations=20, learning_rate=1e-1, depth=6, loss_function='Logloss')
model.fit(X_train, y_train)
preds_class = model.predict(X_test)

s2v_res = np.mean(preds_class == y_test)

print('Gradient boosting via catboost gives accuracy of {:.2f} when applied to sent2vec'.format(s2v_res))
