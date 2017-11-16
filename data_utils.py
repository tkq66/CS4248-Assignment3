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


def split_cross_validation_class_reference(k, training_class_reference):
    """Split the dataset reference into training and validation dataset k times.

    For any number of k-cross validation, split different unique group of datasets
    into training and validation sets. The percentage of data to be held out for
    validation for any k is (1 / k) * 100%.

    Args:
        k:                        A number of cross-validation to be performed.
        training_class_reference: A dict collection of list of file names to the
            text data for each class. Formatting as follows:
                {
                    'class_name': [
                        'text_file_input_path',
                        'text_file_input_path',
                        ...
                    ],
                    ...
                }

    Returns:
        A tuple of training dataset, then validation dataset, both had been splitted
        k number of times. The format for both is a list of class data reference, each
        element in the list represents the k-th cross-vaidation. Formatting as follows:
            [
                {
                    'class_name': [
                        'text_file_input_path',
                        'text_file_input_path',
                        ...
                    ],
                    ...
                },
                ...
            ]

    """
    training_reference = validating_reference = [{}] * k
    for class_name in training_class_reference:
        # Split the item counts for each k iterations
        items_in_class = len(training_class_reference[class_name])
        items_per_k = items_in_class // k
        items_per_k_list = [items_per_k] * (k - 1)
        final_item_count = items_in_class - (items_per_k * (k - 1))
        items_per_k_list.append(final_item_count)

        # Split the validation and training data reference for each k for a class
        for i in range(k):
            val_begin = sum(items_per_k_list[0:i])
            val_end = val_begin + items_per_k_list[i]
            val_index_range = set(range(val_begin, val_end))
            validating_reference[i][class_name] = training_class_reference[class_name][val_begin:val_end]
            training_reference[i][class_name] = [ref for i, ref in enumerate(training_class_reference[class_name]) if i not in val_index_range]
        return training_reference, validating_reference
