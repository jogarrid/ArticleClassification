How to run the code?

web_crawler: contains the scrapping of Web of Science to get paper information (title, author, co-authors, publication type, abstract, title etc.) and creates fired.csv and not_fired.csv.

data_analysis: contains a notebook for the preliminary study of the dataset

Network.py: creates the file network.pd to visualize relations between fired and retained people, that then is used by LoadData.py to create 2 dataframes with the vector representation of the concatenation of titles per author, the concatenation of sources (journal) per author and network information.

loadData.py: By using the functions in helpers.py and in textRepresentations.py, creates the csv files data_preprocessed.csv and network.csv. data_preprocessed.csv contains a list of authors and different vectorial representations of what they have written about (titles of their publications; eg, TITLE W2V) and of for which source they have written (name of the Journals in which they published; eg, SOURCE W2V). It also contains additional information for each author that is used for classification (number of collaborators fired & not fired, number of publications). This data is ready to be used for training and testing of the various classification models. The file network.csv associates each author to all their colaborators, namely all other authors that they have written a paper with.  

Classification_model: contains various machine learning algorithms tested on the dataset to predict whether or not someone can be fired based on their publications and other information


