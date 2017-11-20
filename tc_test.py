"""Text classification testing module.

Running this module will make predictionson a bunch of text files using a 'model' file
before exporting a file labeling each input file with a class. This module requires a
file listing stopwords and a file listing test input files locations.
Run this module with:
    python tc_test.py stopword-list model test-list test-class-list

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import get_stop_word_set, get_testing_reference_blind, get_dict_from_file
from porter import PorterStemmer
from sys import argv
from TextClassifier import TextClassifier


def output_prediction_to_file(predictions, file_name):
    with open(file_name, "w") as fp:
        for input_file_name in predictions:
            class_prediction = predictions[input_file_name]
            fp.write("{} {}\n".format(input_file_name, class_prediction))


def main():
    """Perform model prediction.

    Performs prediction on a list of input texts, outputting classification report.
    Required three command line arguments:
        1.) path to the stopword file
        2.) path to the model file
        3.) path to the reference file
        4.) output file name
    """
    stop_word_file_name = argv[1]
    model_file_name = argv[2]
    testing_list_file_name = argv[3]
    output_file_name = argv[4]

    testing_reference_list = get_testing_reference_blind(testing_list_file_name)
    model = get_dict_from_file(model_file_name)
    class_names = list(model["__bias__"].keys())
    text_classifier = TextClassifier(class_names=class_names,
                                     stemmer=PorterStemmer(),
                                     stopwords=get_stop_word_set(stop_word_file_name),
                                     weights=model)
    predicted_classes = text_classifier.predict(testing_reference_list, activation_fn="step", verbose=True)
    output_prediction_to_file(predicted_classes, output_file_name)


if __name__ == "__main__":
    main()
