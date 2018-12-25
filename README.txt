How to run the code?

web_crawler: contains the scrapping of Web of Science to get paper information (title, author, co-authors, publication type, abstract, title etc.) and creates fired.csv and not_fired.csv.

data_analysis: contains a notebook for the preliminary study of the dataset

Network.py: creates the file network.pd to visualize relations between fired and retained people, that then is used by LoadData.py to create 2 dataframes with the vector representation of the concatenation of titles per author, the concatenation of sources (journal) per author and network information.

LoadData.py: creates data_train.csv and data_test.csv

ML_model: contains various machine learning algorithms tested on the dataset to predict whether or not someone can be fired based on their publications

DL_model: contains deep learning model


