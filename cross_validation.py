"""A module performing k-fold cross-validation on the TextClassifier.

Running this module will performs k-fold cross validation on the text classifier
class. The module will report validation accuracy for each k as well as an average
over all k's before outputting it to a file.

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import get_stop_word_set, get_training_class_reference
import json
from porter import PorterStemmer
from sys import argv
from TextClassifier import TextClassifier as tc


def main():
    """Perform k-fold cross validation.

    Performs k-fold cross validation on the text classifier, reporting on validation
    accuracy for each k and an average over all k's.
    Required three command line arguments:
        1.) k-th times to perform the cross validation
        2.) output file name
        3.) path to the stopword file
        4.) path to the reference file
    """
    k = argv[1]
    output_file_name = argv[2]
    stop_word_file_name = argv[3]
    train_class_list_file_name = argv[4]


if __name__ == "__main__":
    main()
