# NiaveBayes.py
# A component of: hw6
# (C) Brendan J. Herger
# Analytics Master's Candidate at University of San Francisco
# 13herger@gmail.com
#
# Created on 10/24/14, at 3:07 PM
#
# Available under MIT License
# http://opensource.org/licenses/MIT
#
# *********************************
#
# imports
# *********************************
import collections
import re
import sys

import numpy as np
import pandas as pd


# global variables
# *********************************
import bhUtilties

__author__ = 'bjherger'
__license__ = 'http://opensource.org/licenses/MIT'
__version__ = '1.0'
__email__ = '13herger@gmail.com'
__status__ = 'Development'
__maintainer__ = 'bjherger'

# classes
# *********************************

def aggregrate_list_of_dicts(list_of_dicts):

    to_return = collections.Counter()
    for local_dict in list_of_dicts:
        to_return.update(local_dict)
    return to_return

def length_function(list_of_values):
    counter = 0
    for local_list in list_of_values:
        counter += len(local_list)
    return counter

def train(trainDF):
    """
    Train based on the files in trainDF.
    :param trainDF: files to train on
    :param labelDF: Dictionary, of form { fileName : category}
    :param fileStringDic: Dictionary, of form {fileName : list of words in file}
    :param catList: list of categories
    :return: A dictionary, of the form { category : {word : count } }
    """



    possible_categories =  trainDF["label"].unique()



    # add files
    trainDF["cleaned_text"] = trainDF["text"].apply(lambda text: bhUtilties.splitAndCleanString(text))
    trainDF["counter"] = trainDF["cleaned_text"].apply(lambda text: collections.Counter(text))

    # create a new data frame with group by data
    combined_list = list()
    for (df_grouby_name, df_groupby_value) in trainDF.groupby("label"):

        # combined counter for all documents of same label
        aggregrated_counter = aggregrate_list_of_dicts(df_groupby_value["counter"])

        # create dict that contains pandas columns
        local_dict = dict()

        # number of non-unique words
        local_dict["num_non_unique_words"] = length_function(df_groupby_value["cleaned_text"])

        # counter for word frequency
        local_dict["counter"] = aggregrated_counter

        local_dict['num_unique_words'] = len(aggregrated_counter.keys())

        # label
        local_dict["label"] = df_grouby_name

        # add to list, which will later be converted to dataframe
        combined_list.append(local_dict)

    df = pd.DataFrame(combined_list)

    return df

def predict(test_data, trained_df):

    # type check
    test_data = pd.DataFrame(test_data)
    trained_df = pd.DataFrame(trained_df)

    # variables
    total_non_unique_words = trained_df['num_unique_words'].sum()

    # set up test_data
    test_data["cleaned_text"] = test_data["text"].apply(lambda text: bhUtilties.splitAndCleanString(text))
    test_data["counter"] = test_data["cleaned_text"].apply(lambda text: collections.Counter(text))


    # iterate through test data rows (each row is a document)
    guess_list = list()
    for test_data_index, test_data_row in test_data.iterrows():

        test_data_row = test_data_row.to_dict()

        # unpack variables
        local_test_counter = test_data_row['counter']

        # keep track of which is the best label so far
        best_label = None
        best_label_score = None

        # iterate through trained data rows (each row is a label), get score for each label.
        for trained_data_index, trained_data_row in trained_df.iterrows():

            trained_data_row = trained_data_row.to_dict()

            label_num_non_unique_words = trained_data_row['num_non_unique_words']
            label_counter = trained_data_row['counter']
            label_score_list = []

            # iterate through words in test data
            for (word_i, n_i) in local_test_counter.iteritems():

                # number of times word occurs in label
                label_num_occurences = label_counter.get(word_i, 0)

                # probability of word, given label ( +1's for words that were not seen in training)
                p_i = (label_num_occurences + 1.0) / (label_num_non_unique_words + total_non_unique_words + 1.0)

                # create log-scaled label score for word. less negative scores are better
                label_word_score = n_i * np.log(p_i)
                label_score_list.append(label_word_score)

            if sum(label_score_list) > best_label_score:
                best_label = trained_data_row['label']
                best_label_score = sum(label_score_list)
            # print trained_data_row['label']
            # print sum(label_score_list)
        # sys.exit()

        local_dict = dict()
        local_dict['index'] = int(test_data_index)
        local_dict['guess'] = best_label
        guess_list.append(local_dict)

    return_df = pd.DataFrame(guess_list)
    print return_df
    return return_df


class NaiveBayes:

    def __init__(self):
        self.counter = collections.Counter()
        self.trained = None

    def fit(self, data, labels):
        data = pd.DataFrame(data)
        labels = pd.DataFrame(labels)

        labels.columns = ["label"]

        data["label"] = labels

        self.trained = train(data)
        return self


    def predict(self, test_data):
        return predict(test_data, self.trained)



# functions
# *********************************
def main():
    print 'hello world!'


# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    main()
    print 'End Main'

