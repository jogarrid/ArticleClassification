#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gensim
import logging
import pandas as pd


def load_data_training():
    model = gensim.models.KeyedVectors.load_word2vec_format('/content/gdrive/My Drive/GoogleNews-vectors-negative300.bin.gz', binary=True)  

    # naming chosen for same-length, to look pretty
    kicked = pd.read_csv('/content/gdrive/My Drive/MLdata/dismissed_complete.csv')
    stayed = pd.read_csv('/content/gdrive/My Drive/MLdata/nodismissed_complete.csv')

    kicked1 = kicked[['Author','Title paper', 'Journal', 'labels']]
    stayed1 = stayed[['Author','Title paper', 'Journal', 'labels']].sample(frac=(1.0 * kicked.shape[0])/stayed.shape[0]) # random_state = 0

    kicked1 = kicked[['Author','Title paper', 'Journal' ,'labels']]
    stayed1 = stayed[['Author','Title paper','Journal', 'labels']].sample(frac=(1.0 * kicked.shape[0])/stayed.shape[0]) # random_state = 0

    df0 = pd.concat([kicked1, stayed1])

    ix_drop = []
    i1 = 0
    i2 = 1
    iters = 0
    title_papers = df0['Title paper'].tolist()
    journals = df0['Journal'].tolist()
    authors = df0['Author'].tolist()
    labels = df0['labels'].tolist()


    while(i2<len(df0)):
        iters += 1
        if(authors[i2] is authors[i1]):
            title_papers[i1] = title_papers[i1] +" "+ title_papers[i2] #Concatenate strings of the 2 papers this author wrote
            journals[i1] = journals[i1] +" "+ journals[i2] #Concatenate strings of the 2 papers this author wrote

            ix_drop.append(i2-len(ix_drop))
            i2 += 1

        else: 
            i1 = i2
            i2 = i1+1

    for ix in ix_drop: 
        title_papers.pop(ix)
        journals.pop(ix)
        authors.pop(ix)
        labels.pop(ix)


    df0_dict = {'Author': authors, 'Title paper': title_papers, 'Journal': journals, 'labels': labels}
    df0 = pd.DataFrame(df0_dict)
    
    
    #df0 = pd.concat([kicked1, stayed1])
    df1 = df0.copy()
    df1['Author'] = df0['Author'].apply(lambda x: x[3:])
    df1['Label'] = df0['labels'].apply(lambda x: int(x))
    df1['Title paper'] = df0['Title paper'].apply(lambda s: s[1:][:-1])
    df1['Journal'] = df0['Journal'].apply(lambda s: s[1:][:-1])
    df1 = df1.drop(columns=['labels'])
    df1.head()

    # remove trash author names (whose length < 6)
    print('Removing rows:')
    print(df1[df1['Author'].apply(lambda s : len(s) < 6)]['Author'])
    df2 = df1[df1['Author'].apply(lambda s : len(s) >= 6)]
    
    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

        
    
    # In fact we will do classification on "Title paper", so further working on it
    print('Shape of not English words: ', df2[df2['Title paper'].apply(lambda s : not isEnglish(s))].shape)
    df3 = df2.copy()
    df3['Title paper'] = df2['Title paper'].apply(
        lambda s : s.lower()
    ).apply(
        lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s) # delete all not-letters
    ).apply(
        lambda s : re.sub(r"[- + = [ ] - @ & * # |]", " ", s) # substitute defis with spaces
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

    # try to find strange symbols and print them in "Title paper" and print them 
    symbols = df3['Title paper'].apply(
        lambda s: ''.join(c for c in s if not c.isalpha() and c != ' ')
    )
    print(symbols[symbols.apply(lambda s: s != '')])

        # In fact we will do classification on "Title paper", so further working on it
    print('Shape of not English words: ', df2[df2['Journal'].apply(lambda s : not isEnglish(s))].shape)
    df3['Journal'] = df2['Journal'].apply(
        lambda s : s.lower()
    ).apply(
        lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><]", "", s) # delete all not-letters
    ).apply(
        lambda s : re.sub(r"[- + = @ [ ] & * # |]", " ", s) # substitute defis with spaces
    ).apply(
        lambda s : re.sub(r"\d", " ", s) # substitute numbers with spaces
    ).apply(
        lambda s : re.sub(r"\W\w{1,2}\W", " ", s) # naive removal of super-short words
    ).apply(
        lambda s : re.sub(r"\s+", " ", s) # substitute multiple spaces with one
    )
    df3 = df3[df3['Journal'].apply(
        lambda s: s != 'untitled' and s != 'editorial' # drop some common but not-interesting names
    )]
    ​
    # try to find strange symbols and print them in "Title paper" and print them 
    symbols = df3['Journal'].apply(
        lambda s: ''.join(c for c in s if not c.isalpha() and c != ' ')
    )
    print(symbols[symbols.apply(lambda s: s != '')])

    
    df4 = df3.drop(columns=['Author'])
    return df4
    

