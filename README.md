#Naive-Bayes
##Version 2.0
***************************************

(C) Brendan J. Herger
Analytics Master's Candidate at University of San Francisco
13herger@gmail.com

Available under MIT License
http://opensource.org/licenses/MIT


##OVERALL
**********************************

Basically, when fed a list of documents and the categories these files belong to, this program will use a
Naive-Bayes word frequency method to model the categories. This model can then be applied to identify documents of
unknown categories.

This is a simple package for document classification using a Naive-Bayes scheme (fairly well documented, so I won't
describe it here). The main package is fairly independent of I/O, but does require the training data, training labels, 
and test data to be formatted as described in the NaiveBayes documenation. 

An example script is provided in ExampleData.py, but the crux of this module are the following three lines:

    # create model
    nb = NaiveBayes.NaiveBayes()

    # train
    nb.fit(train_df, train_labels)

    # predict
    output = nb.predict(test_df)


##THANK YOU's
**********************************
This base project was created as a project submission for the Summer 2014 section of MSAN 593, taught at the University
of San Francisco by Yannet Interian. I would like to thank her and the faculty for their support and guidance.

It has been updated as an assigment for the Module 2 2014 section of MSAN 621 Machine Learning, taught at the University
of San Francisco by Cynthia 'Cindi' Thompson. I would like to thank her and the faculty for their support and guidance.

As a side note, I've written all code in this package from scratch (excluding imports, obviously). As such, I am able
to make it available under the MIT License.