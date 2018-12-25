import argparse

import pandas as pd
import numpy as np
import scipy
import re
import gensim

from collections import Counter

from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

parser = argparse.ArgumentParser(description='Load training and test data that we use in all the classification models.')

parser.add_argument('-t', '--test_size', type=float, required = False,
                    help='fraction of the data used for test (between 0 and 1)')

args = parser.parse_args()

if(args.test_size == None):
    TEST_SIZE = 0.5
else: 
    TEST_SIZE = args.test_size

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)  

def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def get_mean_w2v_embeddings(titles):
    embs = []
    for title in titles:
        title_emb = np.zeros(300)
        words = title.split(' ')
        for w in words:
            if w in word2vec_model:
                scalar = 1.
#                 scalar = 1. / len(words)
                
                vector = word2vec_model[w]
                
                title_emb += scalar * vector
        embs.append(title_emb)
    return embs

fired = pd.read_csv('data/fired_final.csv')
not_fired = pd.read_csv('data/not_fired_final.csv')

#Eliminate non-letters from author name's
not_fired['author'] = not_fired['author'].apply(lambda s: re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s))

fired = fired.drop('Unnamed: 0', axis = 1)
fired = fired.drop('Unnamed: 0.1', axis = 1)
fired = fired.drop('Authors', axis = 1)
fired = fired.drop('Query', axis = 1)
not_fired = not_fired.drop('Unnamed: 0', axis = 1)

not_fired.columns = ['Abstract', 'Author','Co-authors', 'Heading', 'Keywords', 'Organization',
                    'Publish date', 'Publication type','Source', 'Title paper', 'Label']

ordered = ['abstract', 'author', 'co-authors', 'headings', 'keywords', 'organisation', 
                  'publish_date', 'pubtype', 'source', 'title', 'labels']
fired = fired[ordered]
fired.columns = ['Abstract', 'Author','Co-authors', 'Heading', 'Keywords', 'Organization',
                    'Publish date', 'Publication type','Source', 'Title paper', 'Label']

fired.reset_index(drop = True)

fired1 = fired[['Author','Source','Keywords','Title paper','Label']]
not_fired1 = not_fired[['Author','Source','Keywords','Title paper','Label']]

df0 = pd.concat([fired1, not_fired1])
df0 = df0.reset_index(drop = True)
df1 = df0.copy()
df1['Keywords'] = df0['Keywords'].apply(lambda s: s[1:][:-1])

#1st, we throw away second last names and authors with last names of only 1 letter
authors = []
thrown = []
for i in range(len(df1)):
    #We throw away second last name
    author_l = df1['Author'][i].strip().split(' ')[0:2]
    if(len(author_l) <2 or len(author_l[1]) <2):
        if(author_l not in thrown):
            thrown.append(author_l)
        df1 = df1.drop(i, axis = 0)
    else: 
        author = ' '.join(author_l)
        authors.append(author)
        
df1['Author'] = authors
df1 = df1.reset_index(drop = True)

print('Finished processing the author names')
print('----------------------------------------------------')
#Debugging
#print('We threw away... ' + str(len(thrown)) + " authors because their last name only has 1 letter.")

network_pd = pd.read_csv('data/network.csv')

df2 = df1[df1['Author'].apply(lambda s : len(s) >= 6)]
df3 = df2.copy()

#clean data 
df3['Title paper'] = df2['Title paper'].apply(
    lambda s : s.lower()
).apply(
    lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s) # delete all not-letters
).apply(
    lambda s : re.sub(r"[- + = @ & * # |]", " ", s) # substitute defis with spaces
).apply(
    lambda s : re.sub(r"\d", " ", s) # substitute numbers with spaces
).apply(
    lambda s : re.sub(r"\W\w{1,2}\W", " ", s) # naive removal of super-short words
).apply(
    lambda s : re.sub(r"\s+", " ", s) # substitute multiple spaces with one
)

df3['Source'] = df2['Source'].apply(
    lambda s : s.lower()
).apply(
    lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s) # delete all not-letters
).apply(
    lambda s : re.sub(r"[- + = @ & * # |]", " ", s) # substitute defis with spaces
).apply(
    lambda s : re.sub(r"\d", " ", s) # substitute numbers with spaces
).apply(
    lambda s : re.sub(r"\W\w{1,2}\W", " ", s) # naive removal of super-short words
).apply(
    lambda s : re.sub(r"\s+", " ", s) # substitute multiple spaces with one
)

df3 = df3[df3['Title paper'].apply(
    lambda s: s != 'untitled' and s != 'editorial' # drop some common but not-interesting names
)]

"""
# try to find strange symbols in "Title paper" and in "Source" and print them, for debugging 
symbols = df3['Title paper'].apply(
    lambda s: ''.join(c for c in s if not c.isalpha() and c != ' ')
)
# try to find strange symbols in "Title paper" and print them 
symbols = df3['Source'].apply(
    lambda s: ''.join(c for c in s if not c.isalpha() and c != ' ')
)
"""

print('Finished processing the titles and the sources')
print('----------------------------------------------------')

df4 = df3.drop(columns=['Author'])
df4.head()

titles_num = df4.shape[0]
fired_titles_num = df4[df4['Label'] == 1].shape[0]
notfired_titles_num = df4[df4['Label'] == 0].shape[0]
print('We have ' + str(titles_num) + ' titles, from them ' + str(fired_titles_num) + ' fired and ' + str(notfired_titles_num) + ' not fired')

titles_per_author = {} # author -> article
sources_per_author ={} # author -> source
labels_per_author = {} # author -> label

for i, r in df3.iterrows():
    author = r['Author']
    title = r['Title paper']
    label = int(r['Label'])
    source = r['Source']
    
    titles_per_author[author] = titles_per_author.get(author, '') + ' ' + title # do concatenation
    sources_per_author[author] = sources_per_author.get(author, '') + ' ' + source # do concatenation
    labels_per_author[author] = label

fired_cnt = 0
for k, v in labels_per_author.items():
    if v == 1: fired_cnt+= 1
        
authors = []
titles = []
sources = []
labels = []
notfired_limit = fired_cnt

for k, v in titles_per_author.items():
    if labels_per_author[k] == 0:
        if notfired_limit > 0: notfired_limit -= 1
        else: continue
    
    authors.append(k)
    titles.append(re.sub(r"\s+", " ", v))    
    labels.append(labels_per_author[k])
    sources.append(re.sub(r"\s+", " ", sources_per_author[k]))
    
# aggregated DataFrame
adf4 = pd.DataFrame(data={'Title paper' : titles, 'Source': sources, 'Label' : labels, 'Author': authors}) 

labels_per_titles = {}
same_duplicate = 0
diff_duplicate = 0

# we assume, that noone meets 3 times, which +- correct
for i, r in adf4.iterrows():
    title = r['Title paper']
    label = int(r['Label'])
    if title in labels_per_titles:
        if labels_per_titles[title] == label:
            same_duplicate += 1
        else:
            diff_duplicate += 1
    else:
        labels_per_titles[title] = label
        
        
# throw away all the duplicates, both same and diff, everyone, who meets > 1 times
counter = Counter(adf4['Title paper'])

adf5 = adf4[
    adf4['Title paper'].apply(
        lambda title: counter[title] == 1
    )
]

adf5= adf5.reset_index(drop = True)

print('We train and test the network with ' + str(adf5.shape[0]) + ' authors')
df = adf5

num_fired_l = []
num_nofired_l = []

for author in df['Author']:
    author = ' '.join(author.strip().split(' ')[0:2])
    row = network_pd.loc[network_pd['Author'] == author]
    #Number fired&not fired connections
    num_fired = row['Connections_fired'].values
    num_nofired = row['Connections_nofired'].values
    if(len(num_fired)>0):
        num_fired_l.append(num_fired[0])
        num_nofired_l.append(num_nofired[0])
    else: 
        num_fired_l.append(0)
        num_nofired_l.append(0)
        print('There was an error in finding the author in the network database... ', author)

df['Num Fired'] = num_fired_l
df['Num not fired'] = num_nofired_l

data_train, data_test = train_test_split(df, test_size= TEST_SIZE) # random_state = 0

X_train_title = data_train['Title paper']
X_train_source = data_train['Source']
y_train = data_train['Label']

X_test_title = data_test['Title paper']
X_test_source = data_test['Source']
y_test = data_test['Label']

print('Train size: ' + str(X_train_title.shape[0]) + ' vs test size: ' + str(X_test_title.shape[0]))

X_train_title_embs = get_mean_w2v_embeddings(X_train_title)
X_test_title_embs  = get_mean_w2v_embeddings(X_test_title)
X_train_source_embs = get_mean_w2v_embeddings(X_train_source)
X_test_source_embs  = get_mean_w2v_embeddings(X_test_source)

data_train.loc[:,'Title vector'] = pd.Series(X_train_title_embs, index=data_train.index)
data_test.loc[:,'Title vector'] = pd.Series(X_test_title_embs, index=data_test.index)

data_train.loc[:,'Source vector'] = pd.Series(X_train_source_embs, index=data_train.index)
data_test.loc[:,'Source vector'] = pd.Series(X_test_source_embs, index=data_test.index)

data_train.to_csv('data/data_train.csv')
data_train.to_csv('data/data_test.csv')

