'''
    This file contains functions to report on cross-validation in the NLTK.
    The main function to call is cross_validate_evaluate(num_folds, featuresets, label_list)
    Usage:  given previously defined featuresets and list of labels
        num_folds = 10  # or 5
        label_list = ['pos','neg']
        cross_validate_evaluate(num_folds, featuresets, label_list)
'''
from nltk.metrics import *
# use NLTK to compute evaluation measures from a reflist of gold labels
#    and a testlist of predicted labels for all labels in a list
# returns lists of precision and recall for each label
def eval_measures(reflist, testlist, label_list):
    #initialize sets
    # for each label in the label list, make a set of the indexes of the ref and test items
    #   store them in sets for each label, stored in dictionaries
    # first create dictionaries
    ref_sets = {}
    test_sets = {}
    # create empty sets for each label
    for lab in label_list:
        ref_sets[lab] = set()
        test_sets[lab] = set()

    # get gold labels
    for j, label in enumerate(reflist):
        ref_sets[label].add(j)
    # get predicted labels
    for k, label in enumerate(testlist):
        test_sets[label].add(k)

    # lists to return precision and recall for all labels
    precision_list = []
    recall_list = []
    #compute precision and recall for all labels using the NLTK functions
    for lab in label_list:
        precision_list.append ( precision(ref_sets[lab], test_sets[lab]))
        recall_list.append ( recall(ref_sets[lab], test_sets[lab]))

    return (precision_list, recall_list)

# This function computes F-measure (beta = 1) from precision and recall
def Fscore (precision, recall):
    return (2.0 * precision * recall) / (precision + recall)

# this function prints precision, recall and F-measure for each label
def print_evaluation(precision_list, recall_list, label_list):
    for index, lab in enumerate(label_list):
        print()
        print(lab, 'precision', precision_list[index])
        print(lab, 'recall   ', recall_list[index])
        print(lab, 'F-score  ', Fscore(precision_list[index],recall_list[index]))


# This function performs the cross-validation, creating classifier models for each fold
#    In each fold, it also applies the model to the reference set, getting a list of predicted labels
#    The resulting final list collects all the reference/gold labels and test/predicted labels
def cross_validate_evaluate(num_folds, featuresets, label_list):
    subset_size = int(len(featuresets)/num_folds)
    # overall gold labels for each instance (reference) and predicted labels (test)
    reflist = []
    testlist = []
    # iterate over the folds
    for i in range(num_folds):
        print('Start Fold', i)
        test_this_round = featuresets[i*subset_size:][:subset_size]
        train_this_round = featuresets[:i*subset_size]+featuresets[(i+1)*subset_size:]
        # train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        # add the gold labels and predicted labels for this round to the overall lists
        for (features, label) in test_this_round:
            reflist.append(label)
            testlist.append(classifier.classify(features))

    print('Done with cross-validation')

    # call the evaluation measures function
    (precision_list, recall_list) = eval_measures(reflist, testlist, label_list)
    print_evaluation (precision_list, recall_list, label_list)
