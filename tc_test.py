"""Text classification testing module.

Running this module will make predictionson a bunch of text files using a 'model' file
before exporting a file labeling each input file with a class. This module requires a
file listing stopwords and a file listing test input files locations.
Run this module with:
    python tc_test.py stopword-list model test-list test-class-list

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import get_stop_word_set, get_training_class_reference
from sys import argv
from TextClassifier import TextClassifier as tc


def main():
    pass


if __name__ == "__main__":
    main()
