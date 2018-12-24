from suds.client import Client
import time
import pandas as pd
import csv
import numpy as np
import re

from scraper_WoS import *
from parsing import *

def pub_of_author(Querystring):
	"""get publications from authors query"""
	results = soap.search(Querystring, offset = 1)
	if results.recordsFound > 100:
		new_records = results.records
		for i in range(int(results.recordsFound/100)):
			time.sleep(0.5)
			research = soap.search(Querystring, offset = (i+1)*100+1)
			time.sleep(0.5)
			new_records = new_records + research.records
		#results = new_records
		results.records = new_records
	return results


def list_dismissed(file_path):
	"""extract dismissed authors and the respective affiliation from text file """
	searchnames_dismiss_auths_query = []
	searchnames_dismiss_auths = []
	searchnames_dismiss_univ = []
	print(file_path)
	for line in open(file_path,'r'): 
		QueryString = line.strip()
		#QueryString_author ="AU="+line.split(',')[1]+' '+line.split(',')[2][0]+"*"
		QueryString_author ="AU="+line.split(',')[1]+' '+line.split(',')[2]
		QueryString_aff = line.split(',')[6].replace('ERSITY','')
		author = line.split(',')[1]+' '+line.split(',')[2]
		searchnames_dismiss_auths.append(author)
		searchnames_dismiss_univ.append(QueryString_aff)
		searchnames_dismiss_auths_query.append(QueryString_author)
	
	#remove header
	searchnames_dismiss_auths_query = searchnames_dismiss_auths_query[1:]
	searchnames_dismiss_auths = searchnames_dismiss_auths[1:]
	searchnames_dismiss_univ = searchnames_dismiss_univ[1:]

	return searchnames_dismiss_auths_query, searchnames_dismiss_auths,searchnames_dismiss_univ

def get_info_authors(QueryList):
	"""Getting information about authors through querying web of science"""
	complete_dataset = pd.DataFrame()
	i=0
	for query in QueryList:
		try:
			print(i)
			if i == 2400: # WOS constraints that each ID can only search for 2500 times. Just set smaller counts to renew the ID.
				soap = WosClient(user= 'SWISS10_reproj', password= 'Welcome#10 ', lite=False)
				soap.connect()
			results = pub_of_author(query)
			time.sleep(0.5)
			Soup = BeautifulSoup(results.records,'lxml')
			dataset = construct_dataset(Soup,searchnames_dismiss_auths[i])
			complete_dataset=complete_dataset.append(dataset,ignore_index=True)
			i+=1
			#auth = auth.append(authors,ignore_index=True)
		except:
			print('Error on loop',query)

	return complete_dataset

def disambiguation_authors(df, dismissed_dataset):
	"""Deleting authors that doen't correspond to the one in the given dataset thanks to affiliation"""
	dismissed_pd = cleaning_dataset(dismissed_dataset)
	result = pd.merge(df, dismissed, how='left', on=['organisation'])
	disambiguate_dismissed_dataset = result[result.Authors==result.author]
	return disambiguation_dismissed_dataset


def cleaning_dataset(dismissed_dataset):
	"""split dataset scarpped with a unique affiliation in each row"""
	dismissed_dataset.organisation.apply(pd.Series).merge(dismissed_dataset, left_index = True, right_index = True)
	dismissed_pd = dismissed_dataset.organisation.apply(pd.Series) \
	.merge(dismissed_dataset, right_index = True, left_index = True) \
	.drop(["organisation"], axis = 1) \
	.melt(id_vars = ['abstract', 'author','co-authors','headings','keywords','publish_date','pubtype','source','title'], value_name = "organisation") \
	.drop("variable", axis = 1)\
	.dropna()
	return dismissed_pd

	
def add_labels(df,category):
	"""adding labels 0 or 1  to the datasets"""
	#non dismissed labels
	if category == 0:
		df['labels'] = np.zeros(len(df))

	#dismissed labels
	if category == 1:
		df['labels'] = np.ones(len(df))
	return df

def count_coauthors(dismissed_pd):
	"""coutn number of unique coauthors in the dataset"""
	co_aut = dismissed_pd['co-authors']
	cnt_coauthors=0
	for l in co_aut:
		cnt_coauthors += len(l)
	return cnt_coauthors, co_aut

def reformat_coauthors(co_aut):
	"""reformat co-authors, removing brackets to prepare the query"""
	coauthors = set()
	for l in co_aut :
	    l = l.split(',')
	    for author in l:
	        if "[" in author :
	            author=author.replace('[','')
	            
	        if "]" in author : 
	            author=author.replace(']','')
	        coauthors.add(author)

	return coauthors

def create_undismissed_query(undismissed):
	"""create queries with format AU=name_author from the list of undismissed people"""
	searchnames_undismiss_auths_query = []
	searchnames_undismiss_auths = []
	searchnames_undismiss_univ = []
	for undismissed_pers in undismissed: 
	    QueryString_author ="AU="+undismissed_pers
	    author = undismissed_pers
	    searchnames_undismiss_auths.append(author)
	    searchnames_undismiss_auths_query.append(QueryString_author)
	    
	#remove header
	searchnames_undismiss_auths_query = searchnames_undismiss_auths_query
	searchnames_undismiss_auths = searchnames_undismiss_auths

	return searchnames_undismiss_auths_query,searchnames_undismiss_auths





