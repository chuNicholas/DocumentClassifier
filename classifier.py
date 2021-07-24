import os
import time
from bs4 import BeautifulSoup
from tkinter import Tk, filedialog
import threading
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np

MAX_THREADS = 4

# folder selection dialog
def folder_selector(title_message):
    # folder selector GUI
    root = Tk()
    root.withdraw()

    file_path = filedialog.askdirectory(title=title_message)

    # if empty file path, close program
    if file_path == "": exit()

    # convert file path to regular string
    file_path = os.path.abspath(file_path)

    root.destroy()

    return file_path

# convert html to text
def html_to_text(text_file):
    text = ""
    # grab all the text with the paragraph tags
    with open(text_file, encoding="utf-8") as fp:
        soup = BeautifulSoup(fp, "lxml")
        for words in soup.find_all("p"):
            text += words.get_text() + " "
    
    return text

# parse all files in folder
def parse_folder(articles, classification, fp):
    # create list of the file names inside the folder
    file_names = [os.path.join(fp, article) for article in os.listdir(fp)]

    # for each file inside the folder, append the text version of the article
    # to the  dictionary holding a list of articles
    # key: label or what it is classified as
    # value: list of the article file's text
    for article in file_names:
        articles[classification].append(html_to_text(article))

# concatenate root path with file or folder 
def f_path(root, f_name):
    return os.path.join(root, f_name)

# create a list of articles. each entry in the list contain text from an article
def articles_text_to_list(dataset, articles):

    # sort the keys (labels)
    keys = sorted(articles)
    i = 0
    # keep track of threads
    threads = []

    # loop while we still have distinct articles (labels)
    while i < len(articles):
        while i < len(articles) and threading.activeCount() <= MAX_THREADS:
            # loop through the selected articles' folder and convert all the html files there to text
            thread = threading.Thread(target=parse_folder, args=(articles, keys[i], f_path(dataset, keys[i])))
            threads.append(thread)
            thread.start()
            i += 1

    # finish threads when they are done running
    for thread in threads:
        thread.join()

    # put the all articles in one array
    articles_list = []
    for key in keys:
        articles_list += articles[key]

    return articles_list

def main():
    # start time
    start = time.time()

    # get training dataset folder
    training_dataset_folder = folder_selector("Select Training Dataset Folder")

    # create a training dataset dictionary
    # key: label or what it is classified as
    # value: list of the text from each article
    # i.e., each list element contains all the text from one article
    training_articles = {folder : [] for folder in os.listdir(training_dataset_folder)}

    # training word counts
    training_cv = CountVectorizer(stop_words="english")
    # get the list of text from the articles
    training_articles_words_list = articles_text_to_list(training_dataset_folder, training_articles)
    # fit and transform the model to the training data
    training_count = training_cv.fit_transform(training_articles_words_list)

    # create training target (labels) for training data
    # since keys were sorted, the targets are alphabetically sorted
    # and the number of targets of each classification
    # is calculated by the total number of articles divided by
    # the number of distinct labels
    training_target = []
    for data in sorted(training_articles):
        training_target.extend([data] * (len(training_articles_words_list) // len(sorted(training_articles))))

    # normalize training data
    # instead of using the raw counts for each article
    # this scales the numbers to match the amount of words in the article
    # so the result is the percentage of time the word appears in the article itself
    training_tf_transformer = TfidfTransformer(use_idf=False)
    training_tf = training_tf_transformer.fit_transform(training_count)

    # get the dataset that needs to be classified by our model
    testing_dataset_folder = folder_selector("Select Testing Dataset Folder")

    # new (test) dataset dictionary to store the text from the test set
    testing_articles = {folder : [] for folder in os.listdir(testing_dataset_folder)}

    # convert the testing dataset files into text and store them in a list
    testing_articles_words_list = articles_text_to_list(testing_dataset_folder, testing_articles)
    # using the training count vectorizer, transform it with the testing set
    testing_count = training_cv.transform(testing_articles_words_list)

    # create testing target (labels) for testing data
    # same as the labels created for the training set
    testing_target = []
    for data in sorted(testing_articles):
        testing_target.extend([data] * (len(testing_articles_words_list) // len(sorted(testing_articles))))

    # normalize testing data using term-frequency transformer
    testing_tf = training_tf_transformer.transform(testing_count)

    # create the classifier, this uses the training set and their respective labels
    # naive bayes classifier
    # clf = MultinomialNB().fit(training_tf, training_target)
    # Support Vector Machine Classifier
    clf = SGDClassifier(). fit(training_tf, training_target)

    # predict the labels for the testing set
    predicted = clf.predict(testing_tf)

    # print results
    # for target, result in zip(testing_target, predicted):
    #     print("Target: {}, Result: {}".format(target, result))

    # print metrics/ results of classification
    print(metrics.classification_report(testing_target, predicted))

    # end time
    end = time.time()

    print("Elapsed time: {:.2f} seconds".format(end - start))

if __name__ == "__main__":
    main()