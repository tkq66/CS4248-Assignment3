"""Text classification training module.

Running this module will train a text classifier from a bunch of text files before
exporting the model into a 'model' file. This module requires a file listing stopwords
and a file listing train input files locations and its associated class.
Run this module with:
    python tc_train.py stopword-list train-class-list model

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
import json
from porter import PorterStemmer
import re
from sys import argv
from TextClassifier import TextClassifier as tc


def get_stop_word_set(file_name):
    """Return a list of stopwords that's read from a file."""
    stop_word_set = set()
    with open(file_name, "r") as fp:
        stop_word_set = set(fp.read().strip().split("\n"))
    return stop_word_set


def get_training_class_reference(file_name, tc_location="/home/course/cs4248/"):
    """Return a training input reference.

    Read the file containing input file location and its corresponding class then
    format it to be ready to be fed into the a training function.

    Args:
        file_name:   Path to the refrence file.
        tc_location: Path to the input training files. (default '/home/course/cs4248/')

    Returns:
        Formatted reference data. Formatting as follows:
            {
                'class_name': [
                    'text_file_input_path',
                    'text_file_input_path',
                    ...
                ],
                ...
            }

    """
    default_location = "/home/course/cs4248/"
    is_replace_file_loc = (tc_location != default_location)
    training_class_reference = {}
    with open(file_name, "r") as fp:
        for line in fp:
            preprocessed_line = line.strip()
            if preprocessed_line == "":
                continue
            preprocessed_line = re.sub(default_location, tc_location, preprocessed_line) if is_replace_file_loc else preprocessed_line
            file_path, class_name = preprocessed_line.split()
            if class_name not in training_class_reference:
                training_class_reference[class_name] = []
            training_class_reference[class_name].append(tuple((file_path, class_name)))
    return training_class_reference


def output_model_to_file(model, file_name):
    """Write the classifier weights out to a json file."""
    with open(file_name, "w") as output_file:
        json.dump(model, output_file)


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
    p = PorterStemmer()
    stop_word_set = get_stop_word_set(stop_word_file_name)
    text_classifier = tc.TextClassifier(class_names=training_class_reference.keys(),
                                        stemmer=p,
                                        stopwords=stop_word_set)
    text_classifier.train(p, stop_word_set, training_class_reference)
    output_model_to_file(text_classifier.get_weights(), output_model_file_name)


if __name__ == "__main__":
    main()
