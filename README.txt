How to run the code?

Download https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
place it in the data folder

Organisation of the code:

web_crawler: contains the scrapping of Web of Science to get paper information (title, author, co-authors, publication type, abstract, title etc.) and creates fired.csv and not_fired.csv.

data_analysis: contains a notebook for the preliminary study of the dataset

loadData.py: By using the functions in helpers.py and in textRepresentations.py, creates the csv files data_preprocessed.csv and network.csv. data_preprocessed.csv contains a list of authors and, associated to each of them, different vectorial representations of what they have written about (titles of their publications; eg, TITLE W2V) and of for which source they have written (name of the Journals in which they published; eg, SOURCE W2V). It also contains additional information for each author that is used for classification (number of collaborators fired & not fired, number of publications). This data is ready to be used for training and testing by the various classification models. The file network.csv associates each author to all their colaborators, namely all other authors that they have written a paper with, and is used to find the number of collaborators fired / not fired. 

Classification_model: contains various .py (SVM.py, neuralNetwork.py, naiveBayes.py, catBoost.py) files, each of them trains and tests a different machine learning algorithm to predict whether someone is fired or not based on the vectorial representation of what they have written about (along with the other parameters already described). For this, it loads data_preprocessed.csv, the file created by executin g loadData.py


