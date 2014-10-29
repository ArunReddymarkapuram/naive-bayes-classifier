"""
coding=utf-8
ExampleData.py
A component of: hw6
(C) Brendan J. Herger
Analytics Master's Candidate at University of San Francisco
13herger@gmail.com

Created on 10/24/14, at 3:15 PM

Available under MIT License
http://opensource.org/licenses/MIT
"""

# imports
# *********************************
import os
import random

import numpy as np
import pandas as pd

import NaiveBayes


# global variables
# *********************************
import bhUtilties

__author__ = 'bjherger'
__license__ = 'http://opensource.org/licenses/MIT'
__version__ = '1.0'
__email__ = '13herger@gmail.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'

# functions
# *********************************


def generate_df(directory):
    """
    Create train and test dataframes from the included data sets.
    :param directory: directory for data.
    :return: train_df, test_df
    :rtype: tuple
    """

    # iterate through directory structure, create dataframe from documents
    category_list = os.listdir(directory)
    file_list = list()
    for category in category_list:
        category_directory = os.path.join(directory, category)
        for local_file in bhUtilties.traverseDirectory(category_directory):
            entry_dic = dict()
            entry_dic["path"] = local_file
            entry_dic["category"] = category
            with open(local_file) as fileOpen:
                entry_dic["text"] = fileOpen.read()
            file_list.append(entry_dic)

    # randomly assign train, test sets
    shuffled = random.sample(file_list, len(file_list))
    cutoff = int(len(shuffled) * .666)
    train_list = shuffled[:cutoff]
    test_list = shuffled[cutoff:]

    # convert to DataFrames
    train_df = pd.DataFrame(train_list)
    test_df = pd.DataFrame(test_list)

    # return
    return train_df, test_df


def main():
    """
    main method
    :return:
    """
    # get data
    train_df, test_df = generate_df("data/review_polarity/txt_sentoken")

    # separate training data into data, train_labels
    train_labels = pd.DataFrame(train_df["category"])
    train_df = train_df["text"]

    # create model
    nb = NaiveBayes.NaiveBayes()

    # train
    nb.fit(train_df, train_labels)

    # predict
    output = nb.predict(test_df)

    # check accuracy
    df = pd.DataFrame()
    df['guess'] = output['guess']
    df['actual'] = test_df['category']

    df['correct'] = df['guess'] == df['actual']

    print df
    print np.mean(df['correct'])

# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    main()
    print 'End Main'