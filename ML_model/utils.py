import re
import pandas as pd
import numpy as np
import scipy
from collections import Counter

def get_cleaned_dataset(df):
    df1 = df.copy()
    df1['title'] = df0['title'].apply(
	lambda s : s.lower()
    ).apply(
	lambda s : re.sub(r"[.,/()?:'%\";\[\]!\{\}><\\_]", "", s) # delete all not-letters
    ).apply(
	lambda s : re.sub(r"[- + = @ & * # |]", " ", s) # substitute defis with spaces
    ).apply(
	lambda s : re.sub(r"\d", " ", s) # substitute numbers with spaces
    ).apply(
	lambda s : re.sub(r"\W\w{1,2}\W", " ", s) # naive removal of super-short words
    ).apply(
	lambda s : re.sub(r"\s+", " ", s) # substitute multiple spaces with one
    )
    df1 = df1[df1['title'].apply(
	lambda s: s != 'untitled' and s != 'editorial' # drop some common but not-interesting names
    )] 

    return df1

def get_title_label_dataset(df):
    df1 = df.copy()
    df1 = df1.drop(columns=['author'])

    return df1

def get_cleaned_concatenated_titles_label_dataset(df):
    titles_per_author = {} # author -> article
    labels_per_author = {} # author -> label

    for i, r in df.iterrows():
        author = r['author'], 'aha'
        title = r['title']
        label = int(r['labels'])
	
        titles_per_author[author] = titles_per_author.get(author, '') + ' ' + title # do concatenation
        labels_per_author[author] = label

    kicked_cnt = 0
    for k, v in labels_per_author.items():
        if v == 1: kicked_cnt+= 1
	    
    print('After aggregation we got ' + str(len(titles_per_author)) + ' authors, from which ' + str(kicked_cnt) + ' were kicked and ' + str(len(titles_per_author) - kicked_cnt) + ' not')
    print('Taking prefix of required size for not-kicked')

    authors = []
    titles = []
    labels = []
    stayed_limit = kicked_cnt

    for k, v in titles_per_author.items():
        if labels_per_author[k] == 0:
            if stayed_limit > 0: stayed_limit -= 1
            else: continue
	
        authors.append(k)
        titles.append(re.sub(r"\s+", " ", v))    
        labels.append(labels_per_author[k])
	
    # aggregated DataFrame
    adf = pd.DataFrame(data={'title' : titles, 'labels' : labels})
    print('Got aggregated dataset of size ' + str(len(authors)))

    # to clean the data, let's throw away all the duplicates at all, both same and diff, everyone, who meets > 1 times
    counter = Counter(adf['title'])
    adf1 = adf[adf['title'].apply(lambda title: counter[title] == 1)]

    print('New dataset size after duplicates removal is ' + str(adf1.shape[0]))
    return adf1
