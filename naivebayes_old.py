# naivebayes.py
# (C) Brendan J. Herger
# Analytics Master's Candidate at University of San Francisco
# 13herger@gmail.com
#
# Available under MIT License
# http://opensource.org/licenses/MIT
#
# *********************************
__author__ = 'bjherger'

# imports
############################################

#mine
import bhUtilties

#others
import argparse
import operator
import os
import numpy as np
import pandas as pd
import random
import sys

#specific
from collections import Counter

#import setup
# random.seed(0)  #for consistency. Remove this if you're interested in acutal testing

#variables
############################################

NUM_TRIALS = 5     # Number of trials to run
WANT_PICKLE = True  # Writes dictionary of all file contents to pickle on first run.
                    # Reads from that after (instead of re-loading)
WANT_PRINT = True
PRINT_TO_THIS_FILE = None#"output.txt"

if not WANT_PRINT:
    sys.stdout = open(os.devnull, "w")

if PRINT_TO_THIS_FILE:
    sys.stdout = open(PRINT_TO_THIS_FILE, 'w')


np.seterr(all = 'ignore')
separator = "\n*****************************"

# functions
############################################


def parseArgument():
    """
    Parse command line arguments provided in the form -k value
    :return: A dicitonary, of the form { K : value}
    :rtype: dict
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def genCategoryList(directory):
    return os.listdir(directory)


def getLookupDic(directory, categoryList):
    """
    *This implementation is specific to sub folders pos and neg within the directory*
    Recursively traverse the neg and pos subfolders of the given directory, and return the full path to any files
    contained within these subfolders
    :param directory: directory containing 'neg' and 'pos' subfolders
    :param subfolderList: list of subfolders to be searched separately. This likely maps to categories
    :return: A dictionary, of the form { fileName: category}
    :rtype: dict
    """
    lookupDic = {}
    for category in categoryList:
        categoryDirectory = os.path.join(directory, category)
        for file in bhUtilties.traverseDirectory(categoryDirectory):
            lookupDic[file] = category
    return lookupDic


def getFileStrings(directory, fileList):
    """
    Opens all files, reads them, cleans them, and stores them in the returned dictionary
    *Note: This is unsafe for large databases. In this case it is better to read in files one at a time (and perhaps line by line)*
    :param fileList:  list of files to be read
    :return: A dictionary, of the form { fileName: list of words in file}
    :rtype: dict
    """
    pickleFile = "pickleIt_" + str(hash(directory)) + ".pkl"

    if WANT_PICKLE:
        unpickle = bhUtilties.loadPickle(pickleFile)
        if unpickle:
            return unpickle

    fileStringDic = {}
    for filename in fileList:
        with open(filename) as fileOpen:
            fileOpen = fileOpen.read()
            wordList = bhUtilties.splitAndCleanString(fileOpen)
            fileStringDic[filename] = Counter(wordList)

    bhUtilties.savePickle(fileStringDic, pickleFile)
    return fileStringDic


def createTrainTestLists(fileList, splitPerc=.666):
    """
    Separate fileList into two lists, using draw without replacement. This will result in:
        1) traininglist, of length len(fileList) * splitPerc
        2) testList, or length len(fileList) - len(traininglist)
    :param fileList: list of files to be divided
    :param splitPerc: percentage to use for trainingList.
    :return: (trainList, testList)
    :rtype: tuple
    """
    shuffled = random.sample(fileList, len(fileList))
    cutoff = int(len(shuffled) * splitPerc)
    trainList = shuffled[:cutoff]
    testList = shuffled[cutoff:]
    return (trainList, testList)


def train(trainList, lookupDic, fileStringDic, catList):
    """
    Train based on the files in trainList.
    :param trainList: files to train on
    :param lookupDic: Dictionary, of form { fileName : category}
    :param fileStringDic: Dictionary, of form {fileName : list of words in file}
    :param catList: list of categories
    :return: A dictionary, of the form { category : {word : count } }
    :rtype: dict
    """
    # Variables
    fileStringDic = fileStringDic
    catWordCountDic = dict()


    # Instantiate couters for dictionaries
    for category in catList:
        catWordCountDic[category] = Counter()

    # add files
    [catWordCountDic[lookupDic[filename]].update(fileStringDic[filename]) for filename in trainList]
    # for filename in trainList:
    #     category = lookupDic[filename]
    #
    #     localCounter = fileStringDic[filename]
    #     catWordCountDic[category].update(localCounter)

    #store length number of non-unique words in each category
    lengthDic = {}

    # catWordCountDic["pos"].update(bhUtilties.POS_WORDS)
    # catWordCountDic["neg"].update(bhUtilties.NEG_WORDS)

    for category in catList:
        lengthDic[category] = float(sum(catWordCountDic[category].values()))


    return (catWordCountDic, lengthDic)


def computeCategory(testList, catWordCountDic, lengthDic, fileStringDic, catList):
    """
    Using the dictionary built with training data, make an educated guess about which category
    each member or he testList belongs to.
    :param testList: files to test
    :param catWordCountDic: A dictionary, of the form {category : { word : count } }
    :param lengthDic: A dictionary, of the form {category: number of non-unique words in training set}
    :param fileStringDic: A dictionary, of the form {fileName : list of words contained in file}
    :param catList: list of categories
    :return: A dicitonary, of the form { fileName : guessed category}
    :rtype: dict
    """

    # variables
    catWordCountDic = catWordCountDic
    lengthDic = lengthDic

    P_c = float(len(catList))**-1 #equal weighting

    V = 0
    for value in lengthDic.itervalues():
        V += value

    fileGuessDic = {}

    #loop through
    for filename in testList:

        categorySums = {}

        for category in catList:

            categorySums[category] = 0

        wordList = []

        fileWordCounter = fileStringDic[filename]

        for category in catList:

            wordCountDic = catWordCountDic[category]

            for (word_i, n_i) in fileWordCounter.iteritems():

                # calculate w_i | c
                n = lengthDic[category]

                count_w_c = wordCountDic.get(word_i, 0)
                p_i = (count_w_c + 1) / (n + V + 1)

                #calculate classScore for word
                classScore = n_i * np.log(p_i)

                # add to tally
                categorySums[category] += classScore

        for (category, sum) in categorySums.iteritems():
            categorySums[category] += P_c

        # make a guess, record it in dictionary
        bestCat = max(categorySums.iteritems(), key=operator.itemgetter(1))[0]
        fileGuessDic[filename] = bestCat

    return fileGuessDic


def getAvePrintStats(trainList, testList, fileGuessDic, lookupDic, catList):
    """
    Print relevant information, return overall accuracy for a single iteration
    :param trainList: list of files used to train
    :param testList: list of files evaluated (for which guesses were made)
    :param fileGuessDic: A dictionary, of the form { fileName : guessed category }
    :param lookupDic: A dictionary of the form {fileName : actual category}
    :param catList: A list of categories that were tested
    :return: the accuracy for this iteration
    :rtype: float
    """

    #variables
    fileGuessDic = dict(fileGuessDic)
    lookupDic = dict(lookupDic)

    num_correct = 0
    num_incorrect = 0
    for category in catList:
        #test doc count
        num_test_docs = 0
        for file in testList:
            if lookupDic[file] == category:
                num_test_docs += 1

        num_training_docs = 0
        for file in trainList:
            if lookupDic[file] == category:
                num_training_docs += 1

        num_correct_docs = 0
        num_incorrect_docs = 0
        for (file, guess) in fileGuessDic.iteritems():
            if guess == category:
                if lookupDic[file] == guess:
                    num_correct_docs += 1
                else:
                    num_incorrect_docs +=1

        string = "num__%s_test_docs: %.0f\nnum_%s_training_docs: %.0f\nnum_%s_correct_docs: %.0f" %(category, num_test_docs,
                category, num_training_docs, category, num_correct_docs)
        print string
        num_correct += num_correct_docs
        num_incorrect += num_incorrect_docs

    accuracy = (float(num_correct) / float(num_correct + num_incorrect) ) * 100
    string = "accuracy: %.0f" % (accuracy)
    print string

    return accuracy

def generate_dataframe(directory, categoryList):
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

    df = pd.DataFrame(file_list)
    return df

def runTrials(directory, numTrials, categoryList):
    """
    Runs a sentiment analysis of the documents provided with the -d argument, using a naive-Bayes method.
    This will use .666 of the documents provided to create a training set, and then test with the reminaing documents.

    This function will return the average accuracy obtained in the test set.

    :param directory: directory to search, with sub-directories specified in categoryList
    :param numTrials: number of trials to iterate through documents with
    :param categoryList: list of categories. These are assumed to be sub-folders of the directory parameter.
    :return: average accuracy of trials
    """

    #get files, file lookup
    lookupDic = getLookupDic(directory, categoryList)
    fileList = lookupDic.keys()  # strip away values for blind testing

    #get strings for all documents
    fileStringDic = getFileStrings(directory, fileList)
    print generate_dataframe(directory, categoryList)

    accuracyList = []

    for i in range(numTrials):
        print "\niteration ", i , ":"

        #create train, test lists
        (trainList, testList) = createTrainTestLists(fileList, splitPerc=.666)

        #train
        (catWordCountDic, lengthDic) = train(trainList, lookupDic, fileStringDic, categoryList)


        #guess category for test list
        fileGuessDic = computeCategory(testList, catWordCountDic, lengthDic, fileStringDic, categoryList)

        #get, print stats
        accuracy = getAvePrintStats(trainList, testList, fileGuessDic, lookupDic, categoryList)
        accuracyList.append(accuracy)

    accAve = np.average(accuracyList)
    accSD = np.std(accuracyList)

    print "\nOverall: "
    string = "\nave_accuracy: %.10f%% \nsd_accuracy: %.1f%%" %(accAve, accSD)
    print string
    return accAve


#main
############################################

if __name__ == "__main__":
    print "Begin Main"

    #start the stopwatch
    bhUtilties.timeItStart(printOff=False)

    #get directory to search
    args = parseArgument()
    directory = args["d"][0]

    #get categories
    catList = genCategoryList(directory)

    #run
    accAve = runTrials(directory, NUM_TRIALS, catList)

    #stop the stopwatch
    bhUtilties.timeItEnd(NUM_TRIALS)

    print "End Main"

