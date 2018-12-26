import argparse
import pandas as pd
import numpy as np
import scipy
import re
import gensim
import os

from helpers import *
from textRepresentations import *

from collections import Counter

import sent2vec # epfl-made 

parser = argparse.ArgumentParser(description='Obtaining all the text representations we use for classification.')

args = parser.parse_args()

print('Data is going to be loaded, text represented as a vector in different ways, and the network of authors built. This could take a while')

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)  

sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('data/wiki_unigrams.bin') # 600 features


print('Finished loading googles word2vec pretrained model')
print('----------------------------------------------------')

fired = pd.read_csv('data/fired.csv')
not_fired = pd.read_csv('data/not_fired.csv')

df0 = pd.concat([fired, not_fired])
df0 = df0.reset_index(drop = True)

df0['Author'] = df0['Author'].apply(lambda s: re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s)) #Eliminate non-letters from author names

df1 = df0.copy()
df1['Keywords'] = df0['Keywords'].apply(lambda s: s[1:][:-1])

#1st, we throw away second last names and authors with last names of only 1 letter
authors = []
thrown = []
for i in range(len(df1)):
    #We throw away second last name
    author_l = df1['Author'][i].strip().split(' ')[0:2]
    if(len(author_l) < 2 or len(author_l[1]) < 2):
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

print("We start building/loading the network")
print('----------------------------------------------------')

if(not os.path.exists('data/network.csv')):
    network_pd = build_network(fired)
else:
    network_pd = pd.read_csv('data/network.csv')

print("Finished building/loading the network")
print('----------------------------------------------------')

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
no_per_author = {}     # author -> number of publications
labels_per_author = {} # author -> label

for i, r in df3.iterrows():
    author = r['Author']
    title = r['Title paper']
    label = int(r['Label'])
    source = r['Source']
   
    titles_per_author[author] = titles_per_author.get(author, '') + ' ' + title # do concatenation
    no_per_author[author] = no_per_author.get(author, 0) + 1
    sources_per_author[author] = sources_per_author.get(author, '') + ' ' + source # do concatenation
    labels_per_author[author] = label

fired_cnt = 0
for k, v in labels_per_author.items():
    if v == 1: fired_cnt+= 1
        
authors = []
titles = []
no_titles = []
sources = []
labels = []
notfired_limit = fired_cnt

#Balance dataset so there is the same number of fired & not fired authors
for k, v in titles_per_author.items():
    if labels_per_author[k] == 0:
        if notfired_limit > 0: notfired_limit -= 1
        else: continue
    authors.append(k)
    titles.append(re.sub(r"\s+", " ", v))    
    labels.append(labels_per_author[k])
    sources.append(re.sub(r"\s+", " ", sources_per_author[k]))
    no_titles.append(no_per_author[k])
    
# aggregated DataFrame
adf4 = pd.DataFrame(data={'Title paper' : titles, 'Source': sources, 'No titles': no_titles, 'Label' : labels, 'Author': authors}) 

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

print('We train and test our classification models with ' + str(adf5.shape[0]) + ' authors')
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
        #For debugging, uncomment below. Normally, there is an error in 3 authors
        #print('There was an error in finding the author in the network database... ', author)

df['Num Fired'] = num_fired_l
df['Num not fired'] = num_nofired_l

X_title = df['Title paper']
X_source = df['Source']
y = df['Label']

X_title_w2v = get_mean_w2v_embeddings(X_title, word2vec_model)
X_source_w2v  = get_mean_w2v_embeddings(X_source, word2vec_model)

X_title_w2vtf = get_tfidf_w2v_embeddings(X_title, word2vec_model)
X_source_w2vtf  = get_tfidf_w2v_embeddings(X_source, word2vec_model)

X_title_s2v = get_sent2vec_embeddings(X_title, sent2vec_model)

df.loc[:,'Title w2v'] = pd.Series(X_title_w2v, index=df.index)
df.loc[:,'Source w2v'] = pd.Series(X_title_w2v, index=df.index)
df.loc[:,'Title w2vtf'] = pd.Series(X_title_w2vtf, index=df.index)
df.loc[:,'Source w2vtf'] = pd.Series(X_title_w2vtf, index=df.index)
df.loc[:,'sent2vec'] = pd.Series(X_title_s2v, index=df.index)

#MIssing, we need to either do or not talk about this on the report
#df.loc[:,'Title tf-idf'] = pd.Series(X_title_tf, index=df.index)
#df.loc[:,'Source tf-idf'] = pd.Series(X_title_tf, index=df.index)

#df.loc[:,'Title s2v'] = pd.Series(X_title_s2v, index=df.index)
#df.loc[:,'Source s2v'] = pd.Series(X_title_s2v, index=df.index)

df.to_csv('data/data_preprocessed.csv')

