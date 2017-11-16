"""Text classification training module.

Running this module will train a text classifier from a bunch of text files before
exporting the model into a 'model' file. This module requires a file listing stopwords
and a file listing train input files locations and its associated class.
Run this module with:
    python tc_train.py stopword-list train-class-list model

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import get_stop_word_set, get_training_class_reference, output_dict_to_file
from porter import PorterStemmer
from sys import argv
from TextClassifier import TextClassifier


def main():
    """Perform training and outputting the model.

    Get the training class reference dict, stemmer, stop word set, and classifier object
    then train the classifier before outputting the classifier weights to a file.
    Required three command line arguments:
        1.) path to the stopword file
        2.) path to the reference file
        3.) output model file name
    """
    stop_word_file_name = argv[1]
    train_class_list_file_name = argv[2]
    output_model_file_name = argv[3]

    training_class_reference = get_training_class_reference(train_class_list_file_name, tc_location="")
    text_classifier = TextClassifier(class_names=list(training_class_reference.keys()),
                                     stemmer=PorterStemmer(),
                                     stopwords=get_stop_word_set(stop_word_file_name))
    text_classifier.train(training_class_reference, epochs=200, verbose=True)
    output_dict_to_file(text_classifier.get_weights(), output_model_file_name)


if __name__ == "__main__":
    main()
