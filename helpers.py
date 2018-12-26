import pandas as pd
import numpy as np
import scipy
import re
import gensim


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def build_network(fired):
    """
    Builds network dataframe, which associates each author in the fired.csv and not_fired.csv database with 
    their connections (authors they have written a paper with). The columns of this dataframe are AUTHOR, 
    CONNECTIONS, CONNECTIONS_ID, CONNECTIONS_LABEL, AND LABEL.

    To build the network, we look only at fired.csv which contains the information related to the papers written
    by the fired authors, and maintain the assumption made throughout this work that an author which is not in the 
    fired.csv database 1) is working within Turkey and 2) was not fired.
    
    The result of executing this functions, network_pd is returned, and also a file network.csv is created
    in the folder data.
    """
    network = {'Author': [], 'Connections':[], 'Connections_id':[],'Connections_label':[],'Label':[]}

    fired_list = fired['Author'].unique()

    #id dictionary that associates name and id for the authors
    ids = {}
    for coauthors_ in fired['Co-authors']: 
        coauthors = coauthors_.split(',')

        #Clean list coauthors of weird characters
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
                network['Connections'].append(coauthors_clean) #An author will always be connected to himself (we take care of this later)
                if(author in fired_list):
                    network['Label'].append(1) #Fired
                else: 
                    network['Label'].append(0) #Not fired

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
    return network_pd

