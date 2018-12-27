import argparse
import numpy as np
import gensim
import logging
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np

import scipy
import gensim
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn import cluster, datasets, mixture
from sklearn.model_selection import cross_val_score


def reduce_dim(method,label,state):
	data_by_label = data[data[method] == label]
	data_flat = data_by_label['Title w2v'].values.flatten()
	data_flat = np.hstack(data_flat).reshape(-1,300)
	pca = PCA(n_components=2)
	#principalComponents_nofired = pca.fit(nofired)
	data_reduced = pca.fit_transform(data_flat)
	print('PCA score 2 first components',method, state ,pca.explained_variance_ratio_)
	return data_reduced

def plot_clustering(method,nofired,fired):
	plt.scatter(nofired[:,0],nofired[:,1],color='blue',label='no fired',alpha=0.1)
	plt.scatter(fired[:,0],fired[:,1],color='red',label='fired',alpha=0.1)
	plt.title(method)
	plt.xlabel('pca1')
	plt.ylabel('pca2')
	plt.legend()
	path_to_save = '../images/'+method
	plt.savefig(path_to_save)


#For reproducibility of results
np.random.seed(100)

parser = argparse.ArgumentParser(description='Implement clustering methods for classification')

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
X_test = np.array([data_test['Title w2v'][i] for i in range(len(data_test))])
y_test = data_test['Label']

#config values
random_state = 170
n_samples=100

#flat test and train set
data_flat_test = data_test['Title w2v'].values.flatten()
data_flat_test = np.hstack(data_flat_test).reshape(-1,300)

data_flat_train = data_train['Title w2v'].values.flatten()
data_flat_train = np.hstack(data_flat_test).reshape(-1,300)

#apply Kmeans clustering
kmeans = KMeans(n_clusters=2, random_state=random_state)
#fit with train set
kmeans.fit(data_flat_train)
#predict with test set to compute accuracy
y_pred = kmeans.predict(data_flat_test)
true_labels=list(data_test.Label)

#compute accuracy
a = y_pred-true_labels
acc = list(a).count(0)/(len(a))
print('KMeans gives accuracy of {:.2f} when applied to w2v mean concatenated titles'.format(acc))


#seperate when label=0 and label=1 in the predicted model on the entire dataset and reduce dim to 2
#flatten the dataset to get a matrix nb_itemsx300 (300 is the length of w2v vectors)
data_flat = data['Title w2v'].values.flatten()
data_flat = np.hstack(data_flat).reshape(-1,300)

#reduce dataset into 2 dimensions
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_flat)
y_pred_total = kmeans.fit_predict(data_reduced)
data['prediction KMEANS'] = y_pred_total
nofired_reduced_kmeans = reduce_dim('prediction KMEANS',0,'nofired')
fired_reduced_kmeans = reduce_dim('prediction KMEANS',1,'fired')

#save the prediction in images folder instead of plot 
plot_clustering('prediction KMEANS',nofired_reduced_kmeans,fired_reduced_kmeans)

#same thing with Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmm.fit_predict(data_flat_train)
y_pred_gmm = gmm.predict(data_flat_test)
data_test['prediction gaussian']=y_pred_gmm


a = y_pred_gmm-true_labels
acc = list(a).count(0)/(len(a))
print(acc)
print('Haussian Mixture Model gives accuracy of {:.2f} when applied to w2v mean concatenated titles'.format(acc))

y_pred_total = gmm.fit_predict(data_reduced)
data['prediction gaussian'] = y_pred_total

nofired_reduced_gmm = reduce_dim('prediction gaussian',0,'no fired')
fired_reduced_gmm = reduce_dim('prediction gaussian',1,'fired')


plot_clustering('prediction gaussian',nofired_reduced_gmm,fired_reduced_gmm)



