"""Module for convenient utility functions.

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
import re


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
            training_class_reference[class_name].append(file_path)
    return training_class_reference
