import argparse
import numpy as np
import gensim
import logging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
# example of training a final classification model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.optimizers import Adam

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

### all vectors are now represented as strings. Represent as arrays of floats.
data['Title w2v'] = data['Title w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2v'] = data['Source w2v'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Title w2vtf'] = data['Title w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])
data['Source w2vtf'] = data['Source w2vtf'].apply(lambda s: [float(char) for char in s.strip('[]').replace('\n', '').split()])

data_train, data_test = train_test_split(data, test_size=TEST_SIZE) # random_state = 0
print('Size of test set: ', str(len(data_test)), ' size of train set: ', str(len(data_train)))
data_train = data_train.reset_index(drop = True)
data_test = data_test.reset_index(drop = True)

X = np.array([data_train['Title w2v'][i]+data_train['Source w2v'][i]+[data_train['Num Fired'][i]]+[data_train['Num not fired'][i]] +[data_train['No titles'][i]] for i in range(len(data_train))])
y = data_train['Label'].values

Xtest = np.array([data_test['Title w2v'][i]+data_test['Source w2v'][i]+[data_test['Num Fired'][i]]+[data_test['Num not fired'][i]] +[data_test['No titles'][i]] for i in range(len(data_test))])
ytest = data_test['Label'].values

scalar = StandardScaler()
scalar.fit(X)
X = scalar.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(2004, input_dim=603, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2, input_shape=(100,)))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
, metrics = ['accuracy'])
model.fit(X, y, epochs=100,batch_size = 50, verbose=2)
ypred = model.predict_classes(Xtest)
acc = sum(ypred[:,0] == ytest)/len(ytest)

print('Using a deep neural netwrok we get an accuracy in the test set of:  ', acc)
