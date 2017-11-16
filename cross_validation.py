"""A module performing k-fold cross-validation on the TextClassifier.

Running this module will performs k-fold cross validation on the text classifier
class. The module will report validation accuracy for each k as well as an average
over all k's before outputting it to a file.

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import get_stop_word_set, get_training_class_reference, output_dict_to_file
from porter import PorterStemmer
from sys import argv
from TextClassifier import TextClassifier


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
    k = int(argv[1])
    output_file_name = argv[2]
    stop_word_file_name = argv[3]
    train_class_list_file_name = argv[4]

    training_class_reference = get_training_class_reference(train_class_list_file_name, tc_location="")
    text_classifier = TextClassifier(class_names=list(training_class_reference.keys()),
                                     stemmer=PorterStemmer(),
                                     stopwords=get_stop_word_set(stop_word_file_name))
    error_report = text_classifier.cross_validate(k,
                                                  training_class_reference,
                                                  epochs=100,
                                                  activation_fn="step",
                                                  lr=0.01,
                                                  verbose=True)
    output_dict_to_file(error_report, output_file_name)


if __name__ == "__main__":
    main()
