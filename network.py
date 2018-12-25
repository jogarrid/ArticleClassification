

import numpy as np
import gensim
import logging
import networkx as nx
import pandas as pd


fired = pd.read_csv('data/fired_final.csv')
not_fired = pd.read_csv('data/not_fired_final.csv')
fired_list = fired['author'].unique()
#Delete random columns to have same number of fired and not fired
indexes_drop = np.arange(len(not_fired)-len(fired))
np.random.shuffle(indexes_drop)
not_fired= not_fired.drop(indexes_drop, axis = 0)

all_authors = pd.concat((fired, not_fired), sort = False)
all_authors = all_authors.reset_index(drop = True)

network = {'Author': [], 'Connections':[], 'Connections_id':[],'Connections_label':[],'Label':[]}

#id dictionary that associates name and id for the authors
ids = {}
for coauthors_ in fired['co-authors']: 
    coauthors = coauthors_.split(',')
    #"Clean list coauthors of weird characters"
    coauthors_clean = []
    for j in range(len(coauthors)-1):
        coauthors_clean.append(' '.join(coauthors[j][2:len(coauthors[j])-1].split(' ')[0:2]))
    j = len(coauthors) - 1
    coauthors_clean.append(' '.join(coauthors[j][2:len(coauthors[j])-2].split(' ')[0:2]))
    #Cleaning of list finished
    for i in range(len(coauthors_clean)):
        author = coauthors_clean[i]
        if(author not in network['Author']):
            network['Author'].append(author)
            ids[author] = len(network['Author']) - 1
            network['Connections'].append(coauthors_clean) #An author will always be connected to himself
            if(author in fired_list):
                network['Label'].append(1) #fired
            else:  #non fired
                network['Label'].append(0) #fired

        else: #Author already part of the network
            #in which index is the author in the network?
            found = False
            j = 0
            while(not found):
                if(network['Author'][j] == author):
                    found = True
                    ix = j
                j += 1
            #index found
            #For each coauthor: add it to the network of the main author
            for coauthor in coauthors_clean:
                if(coauthor not in network['Connections'][ix]):
                    network['Connections'][ix].append(coauthor)
                    
#Add connections_id, id of authors in the connections
errors = 0
network['Connections_id'] = []
network['Connections_label'] = []

for i in range(len(network['Author'])):
    indexes = []
    labels = []
    my_label = network['Label'][i]
    for connection in network['Connections'][i]:
        #Find ix of connection
            ix = ids[connection]
            label = network['Label'][ix]
            indexes.append(ix)
            labels.append(label)

    #remove myself from my connections
    labels.remove(my_label)
    indexes.remove(i)
    network['Connections_id'].append(indexes)
    network['Connections_label'].append(labels)

#Add fields no of connections fired and no of connections not fired 
network['Connections_fired'] = []
network['Connections_nofired'] = []
for i in range(len(network['Author'])):
    fired = sum(network['Connections_label'][i])
    network['Connections_fired'].append(fired)
    network['Connections_nofired'].append(len(network['Connections_label'][i]) - fired)
         
network_pd = pd.DataFrame(network)
network_pd.to_csv('data/network.csv')

