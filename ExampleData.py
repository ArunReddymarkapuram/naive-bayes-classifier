# ExampleData.py
# A component of: hw6
# (C) Brendan J. Herger
# Analytics Master's Candidate at University of San Francisco
# 13herger@gmail.com
#
# Created on 10/24/14, at 3:15 PM
#
# Available under MIT License
# http://opensource.org/licenses/MIT
#
# *********************************
#
# imports
# *********************************
import os
import random
import re
import sys

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


def genCategoryList(directory):
    return os.listdir(directory)


def generate_dataframe(directory):
    categoryList = genCategoryList(directory)
    file_list = list()
    for category in categoryList:
        categoryDirectory = os.path.join(directory, category)
        for file in bhUtilties.traverseDirectory(categoryDirectory):
            entry_dic = dict()
            entry_dic["path"] = file
            entry_dic["category"] = category
            with open(file) as fileOpen:
                entry_dic["text"] =fileOpen.read()
            file_list.append(entry_dic)
    shuffled = random.sample(file_list, len(file_list))
    cutoff = int(len(shuffled) * .666)
    trainList = shuffled[:cutoff]
    testList = shuffled[cutoff:]
    train_df = pd.DataFrame(trainList)
    test_df = pd.DataFrame(testList)
    return train_df, test_df


def main():

    train, test = generate_dataframe("data/review_polarity/txt_sentoken")
    nb = NaiveBayes.NaiveBayes()

    labels = pd.DataFrame(train["category"])
    train.drop("category", 1)


    nb.fit(train, labels)

    output = nb.predict(test)

    df = pd.DataFrame()
    df['guess'] = output['guess']
    df['actual'] = test['category']

    df['correct'] = df['guess'] == df['actual']


    print df
    print np.mean(df['correct'])

# main
# *********************************

if __name__ == '__main__':
    print 'Begin Main'
    main()
    print 'End Main'

