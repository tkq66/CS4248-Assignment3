"""Text classification module, providing a TextClassifier class to train and classify texts.

The text classifier requires that the input be in a specific format (more details on
TextClassifier.train's docstring). The classifier uses perceptron algorithm for training.

Author: Teekayu Klongtruajrok
For CS4248 Assignment 3 - National University of Singapore (NUS) 2017
"""
from data_utils import split_cross_validation_class_reference
from math import exp, sqrt
from random import gauss
import datetime


class TextClassifier:
    """A text classifier that uses perceptron algorithm for classification."""

    __INITIAL_WEIGHT_VALUE = 0
    __INITIAL_BIAS_VALUE = 1
    __BIAS_WEIGHT_KEY = "__bias__"
    __SUPPORTED_ACTIVATION = ["step", "sigmoid"]

    def __init__(self, class_names, stemmer, stopwords, weights=None):
        """Initialize weight dict and take in other dependencies.

        Initialize the object and accept tools for preprocessing: the stemmer object
        for stemming the word and a set of stopwords to ignore during training. The
        'weight' dict represents the classifier model. The dict is in the following format:
            {
                "__bias__": {
                    "class1": 1.1,
                    "class2": 1.4,
                    ...
                },
                "word1": {
                    "class1": 0.6,
                    "class2": 2.1,
                    ...
                }
            }

        Args:
            class_names: A list of all class labels.
            stemmer:     An instance of the stemmer object, must have 'stem' method.
            stopwords:   A set of stopwords (words to ignore during training).
            weights:     A weight model object, to populate the model from a checkpoint. (default None)
        """
        assert isinstance(class_names, list)
        assert all(isinstance(name, str) for name in class_names)
        assert hasattr(stemmer, "stem")
        assert isinstance(stopwords, set)
        assert all(isinstance(word, str) for word in stopwords)

        self.__weights = weights if weights is not None else {self.__BIAS_WEIGHT_KEY: {name: self.__INITIAL_BIAS_VALUE for name in class_names}}
        self.__class_names = class_names
        self.__stemmer = stemmer
        self.__stop_word_set = stopwords

    def get_weights(self):
        """Return classifier weight model."""
        return self.__weights

    def train(self, training_class_reference, epochs, activation_fn="step", lr=0.01, validate_data=None, validate=False, verbose=False, basic_log=False):
        """Train perceptron classifier from an input dictionary.

        Train a text classifier using perceptron algorithm from a given input data.
        Model is represented by the 'weight' variable, attained with 'getWeights' method.

        Args:
            training_class_reference: Input dictionary data to train the model, complying
                to the following format:
                {
                    'class_name': [
                        'text_file_input_path',
                        'text_file_input_path',
                        ...
                    ],
                    ...
                }
            epochs:                   A number of iterations over the dataset to train.
            activation_fn:            A string identifier of the activation function to use. (default 'step')
                                        Supported activation functions:
                                            1.) "step" - Step function
                                            2.) "sigmoid" - Sigmoid function
            lr:                       A floating point learning rate. (default 0.01)
            validate:                 A boolean switch whether to perform validation per epoch or not. (default False)
            verbose:                  A boolean switch whether to print training details or not. (default Fasle)
            basic_log:                A boolean switch whether to printn basic logging or not. (default False)
        """
        assert isinstance(training_class_reference, dict)
        assert all(isinstance(training_class_reference[c], list) for c in training_class_reference.keys())
        assert all(all(isinstance(n, str) for n in training_class_reference[c]) for c in training_class_reference.keys())
        assert activation_fn in self.__SUPPORTED_ACTIVATION
        assert lr > 0
        assert epochs > 0

        # TODO CLEAN UP THIS HACK
        self.__training_class_reference = training_class_reference

        if verbose:
            print("Begin training...")
        for epoch in range(epochs):
            print(datetime.datetime.now())
            for index, training_class_name in enumerate(training_class_reference):
                acc = 0.0
                data_count = 0
                # Train with all the available data, specifying 1 when it's the correct class and -1 otherwise
                for index, data_class_name in enumerate(training_class_reference):
                    expected_output = 1 if data_class_name == training_class_name else 0
                    for i in range(len(training_class_reference[data_class_name])):
                        file_path = training_class_reference[data_class_name][i]
                        text_vector = self.__get_text_vector_from_file(file_path)
                        # Send new input through forward pass for the class currently training
                        result, affine_result, text_reference_counter = self.__forward_pass(training_class_name, text_vector, activation_fn)
                        binary_result = self.__predict_binary(result, activation_fn)
                        # Update the weights for the class currently training
                        self.__update_weights(training_class_name, expected_output, text_reference_counter, binary_result, lr)
                        # Caclculate the error
                        # if validate:
                        #     acc += 1 if self.__predict_multi(text_vector, activation_fn) == training_class_name else 0
                        #     data_count += 1
                        if verbose:
                            print("Progress... {}/{}\033[K".format(i+1, len(training_class_reference[data_class_name])), end="\r")
                # for i in range(len(training_class_reference[training_class_name])):
                #     file_path = training_class_reference[training_class_name][i]
                #     text_vector = self.__get_text_vector_from_file(file_path)
                #     # Send new input through forward pass for the class currently training
                #     result, affine_result, text_reference_counter = self.__forward_pass(training_class_name, text_vector, activation_fn)
                #     binary_result = self.__predict_binary(result, activation_fn)
                #     # Update the weights for the class currently training
                #     self.__update_weights(training_class_name, 1, text_reference_counter, binary_result, lr)
                #     # Caclculate the error
                #     sse += (binary_result - 1) ** 2
                #     acc += 1 if self.__predict_multi(text_vector, activation_fn) == training_class_name else 0
                #     data_count += 1
                #     if verbose:
                #         print("Progress... {}/{} - sse: {}\033[K".format(i+1, len(training_class_reference[training_class_name]), sse), end="\r")
                # Move on to training the next class if there is no more error
                # if validate:
                #     acc /= data_count
                if verbose:
                    print("\nTraining class {} - Epoch {}/{} - acc: {}".format(training_class_name, epoch+1, epochs, acc))
                if basic_log:
                    print("Training class {} - Epoch {}/{}\033[K".format(training_class_name, epoch+1, epochs), end="\r")
            self.__validate(validate_data[0], validate_data[1], validate_data[2], validate_data[3], validate_data[4])

    def predict(self, input_data_reference, activation_fn="step", verbose=False):
        """Predict class from an input data.

        Take in multiple input file names, read the files, and label them with the correct class.

        Args:
            input_data_reference: A list of file names referncing the text data file.
            activation_fn:        A string identifier of the activation function to use. (default 'step')
                                    Supported activation functions:
                                        1.) "step" - Step function
                                        2.) "sigmoid" - Sigmoid function
            verbose:              A boolean switch whether to print details or not. (default Fasle)

        Returns:
            A dict of file names and its associated class label. Formatting as follows:
                {
                    "some file path": "some labeled class",
                    ...
                }

        """
        assert isinstance(input_data_reference, list)

        if verbose:
            print("Begin prediction...")
        total_input = len(input_data_reference)
        results = {}
        for i, file_path in enumerate(input_data_reference):
            if verbose:
                print("Predicting data... {}/{}\033[K".format(i, total_input), end="\r")
            # Get raw data
            text_vector = self.__get_text_vector_from_file(file_path)
            predicted_class_name = self.__predict_multi(text_vector, activation_fn)
            results[file_path] = predicted_class_name
        return results

    def cross_validate(self, k, training_class_reference, epochs, activation_fn="step", lr=0.01, verbose=False):
        """Perform k-fold cross validation from a given dataset.

        Performs training and validation on the given dataset k times. The dataset will first be splitted
        into training and validating dataset, uniquely for k number of times. Then for each k-th entry,
        train on the splitted training dataset and validate on the splitted validating dataset. Finally,
        the result of each validation is returned.

        Note: This is purely for reporting performance of the machine, and running this will not produce
        a persistent weight model. To produce a weight model after training, consult the "train" method.

        Args:
            k:                        A number of cross-validation to be performed.
            training_class_reference: Input dictionary data to train the model, complying
                to the following format:
                {
                    'class_name': [
                        'text_file_input_path',
                        'text_file_input_path',
                        ...
                    ],
                    ...
                }
            epochs:                   A number of iterations over the dataset to train.
            activation_fn:            A string identifier of the activation function to use. (default 'step')
                                        Supported activation functions:
                                            1.) "step" - Step function
                                            2.) "sigmoid" - Sigmoid function
            lr:                       A floating point learning rate. (default 0.01)
            verbose:                  A boolean switch whether to print details or not. (default Fasle)

        Returns:
            An accuracy performance report object for each k, and an average over all k's. Formatting as follows:
                {
                    "1": 0.94532,
                    "2": 0.95323,
                    ... ,
                    "avg": 0.94623
                }

        """
        avg_label = "avg"
        error_report = {avg_label: 0}
        training_reference, validating_reference = split_cross_validation_class_reference(k, training_class_reference)
        if verbose:
            print("Begin {}-fold cross-validation".format(k))
        for i in range(k):
            # Referesh the model
            k_label = str(i)
            self.__clear_weights()
            raw_validation_data = [file_path for class_name in validating_reference[i] for file_path in validating_reference[i][class_name]]
            solution_validation_data = {file_path: class_name for class_name in validating_reference[i] for file_path in validating_reference[i][class_name]}
            if verbose:
                print("K-th {}/{}".format(i+1, k))
            self.train(training_reference[i], epochs, activation_fn=activation_fn, lr=lr, validate_data=(k, raw_validation_data, solution_validation_data, activation_fn, True), validate=True, verbose=True, basic_log=False)
            error_report[k_label] = self.__validate(k, raw_validation_data, solution_validation_data, activation_fn, verbose=False)
            error_report[avg_label] += error_report[k_label]
        error_report[avg_label] /= k
        return error_report

    def __validate(self, k, raw_validation_data, solution_validation_data, activation_fn, verbose=False):
        predicted_classes = self.predict(raw_validation_data, activation_fn=activation_fn, verbose=verbose)
        # Score the prediction
        if verbose:
            print("Scoring the prediction...")
        error_report = 0
        for file_path in predicted_classes:
            isCorrect = predicted_classes[file_path] == solution_validation_data[file_path]
            error_report += 1 if not isCorrect else 0
        error_report /= len(raw_validation_data)
        if verbose:
            print("Error rate: {}".format(error_report))
        return error_report

    def __clear_weights(self):
        """Clean up the weight model, leaving only default init values."""
        self.__weights = {self.__BIAS_WEIGHT_KEY: {name: self.__INITIAL_BIAS_VALUE for name in self.__class_names}}

    def __get_text_vector_from_file(self, file_name):
        """Read the text from file, remove all the spaces, and split into a vector.

        Args:
            file_name: File path string to the input text file.
        """
        text_vector = []
        with open(file_name, "r", encoding="iso-8859-1") as fp:
            text_vector = fp.read().split()
        return text_vector

    def __forward_pass(self, class_label, text_vector, activation_fn):
        """Pass the raw text input vector through various transformations.

        Take the raw input text vector then preprocess and normalize it. Then pass
        the normalized text vector through:
            1. Affine transformations
            2. Step-function activation
        The numeric output should be close to target output if the model have high
        predictive power of this dataset.

        Args:
            class_label:   A string label for the class to be evaluated.
            text_vector:   A list of raw text input.
            activation_fn: A string identifier of the activation function to use.

        Returns:
            A number that results from the forward pass transformation, an indicator
            of how successful the model is in predicting the current class.

        """
        text_reference_counter = self.__preprocess_text(text_vector, class_label)
        affine_result = self.__affine_transformation(class_label, text_reference_counter)
        result = self.__activation(affine_result, activation_fn)
        return result, affine_result, text_reference_counter

    def __update_weights(self, class_label, expected_output, text_reference_counter, network_output, lr):
        """Update the weights of the model, following the perceptron updating rule.

        Update each word-feature's weight for a specified class by the percetron update rule:
            weight_word_n+1 = weight_word_n + (lr * (target_out - network_out) * word)
        Since one weight model is associated with only one class, this  one-hot encoding
        of the target output implies that target_out must be 1.

        Args:
            class_label:            A string label for the class to have the weight updated.
            expected_output:        A number either 1 or -1, guding the network_output.
                                        1 means this is the correct class.
                                        -1 means this is the wrong class.
            text_reference_counter: A dict of words with its associated word-count.
            network_output:         An output number from the perceptron network.
            lr:                     A floating point learning rate.
        """
        for word in text_reference_counter:
            if word not in self.__weights:
                continue
            direction = 1 if network_output == expected_output else -1
            delta = lr * direction * text_reference_counter[word]
            self.__weights[word][class_label] += delta

    def __predict_multi(self, text_vector, activation_fn):
        """Predict multiple classes by finding the class with max activation value.

        Args:
            text_vector:   A list of input text.
            activation_fn: A string identifier of the activation function to use.

        Returns:
            Predicted class name.

        """
        predicted_classes = []
        # Calculate the forward pass for for all perceptron classes, then select largest one
        for class_name in self.__class_names:
            # Send new input through forward pass
            result, affine_result, text_reference_counter = self.__forward_pass(class_name, text_vector, activation_fn)
            predicted_classes.append(tuple((class_name, affine_result)))
        predicted_class_name, _ = max(predicted_classes, key=lambda x: x[1])
        return predicted_class_name

    def __predict_binary(self, value, fn):
        """Predict the binary class based on different activation function type.

        Args:
            value: A number input.
            fn:    A string identifier of the activation function to use.

        Returns:
            The result of the activation function.

        """
        if fn == "step":
            return 1 if value > 0 else 0
        elif fn == "sigmoid":
            return 1 if value >= 0.5 else 0
        else:
            print("Invalid activation function provided: {}, falling back on to step activation.".format(fn))
            return 1 if value >= 0 else 0

    def __preprocess_text(self, text_vector, class_label):
        """Preprocess a text vector.

        Preprocess a text vector (list of texts/tokens) following these steps:
            1. Preprocess the word.
            2. Initialize dict entry in 'weight' for unknown words.
            3. Count the words.
            4. Normalize the word frequency (word_count / total_word_count).

        Args:
            text_vector: Raw unprocessed list of text/tokens.

        Returns:
            A dictionary of words with its normalized frequency. For example:
                {
                    "word1": 0.75, <-- normalized count
                    "word2": 0.25
                }

        """
        # Initialize the counter reference for each word, with an auxiliary bias identity modifier
        text_reference_counter = {self.__BIAS_WEIGHT_KEY: 1}
        total_word_count = len(text_vector)
        for i in range(total_word_count):
            word = self.__preprocess_word(text_vector[i])
            if word is None:
                continue
            # Initialize weight for word not yet seen
            if word not in self.__weights:
                # correlation_coeff = self.__calculate_chi_squared(word, class_label)
                # if correlation_coeff >= 1:
                #     self.__weights[word] = {name: self.__INITIAL_WEIGHT_VALUE for name in self.__class_names}
                self.__weights[word] = {name: self.__INITIAL_WEIGHT_VALUE for name in self.__class_names}
            # Increment total word counts
            text_reference_counter[word] = text_reference_counter.get(word, 0) + 1
        # Normalize the word frequency so that texts of different length will
        # contribute to the machine in the same scale
        # for word in text_reference_counter:
        #     if word == self.__BIAS_WEIGHT_KEY:
        #         continue
        #     text_reference_counter[word] /= total_word_count
        return text_reference_counter

    def __preprocess_word(self, token):
        """Preprocess a word/token.

        Preprocess a raw word following these steps:
            1. Convert token to lower case if token is a word, otherwise leave it alone.
            2. Refuse to proceed and return 'None' if the word is in a list of stopwords.
            3. Stem the word if token is a word, otherwise leave it alone.

        Args:
            token: Unprocessed input token.

        Returns:
            The preprocessed word if the word is not a stop word, otherwise return 'None'.

        """
        # if token.isnumeric():
        #     return None
        # Normalize the text to lowercase
        input_word = token.lower() if token.isalpha() else token
        # Ignore stopwords
        if input_word in self.__stop_word_set:
            return None
        # Stem the word if it is indeed a word
        word = self.__stemmer.stem(input_word, 0, len(input_word) - 1) if token.isalpha() else input_word
        return word

    def __calculate_chi_squared(self, stemmed_word, class_label):
        n00 = 0
        n01 = 0
        n10 = 0
        n11 = 0
        for class_name in self.__training_class_reference:
            for file_path in self.__training_class_reference[class_name]:
                text_vector = self.__get_text_vector_from_file(file_path)
                total_word_count = len(text_vector)
                processed_words = set()
                for i in range(total_word_count):
                    word = self.__preprocess_word(text_vector[i])
                    if word is None:
                        continue
                    processed_words.add(word)
                have_word = stemmed_word in processed_words
                in_class = class_name == class_label
                if have_word and in_class:
                    n11 += 1
                elif have_word and not in_class:
                    n10 += 1
                elif not have_word and in_class:
                    n01 += 1
                elif not have_word and not in_class:
                    n00 += 1
        # print(stemmed_word, n00, n11, n10, n01)
        num = ((n00 + n01 + n10 + n11) * (((n11 * n00) - (n10 * n01)) ** 2))
        den = ((n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00))
        if den == 0:
            return 0
        return num / den

    def __affine_transformation(self, class_label, text_reference_counter):
        """Calculate the affine transformation of a given input for a given class.

        Perform affine transformation calculation on a text vector with the weight of
        a given class. Affine transformation is in the following form:
            bias + (w_1 * word_1) + (w_2 * word_2) + ... + (w_n * word_n)

        Args:
            class_label:            A string label for the class to reference the weights.
            text_reference_counter: A dict of words with the value being the normalized frequency.

        Returns:
            The result of the affine transformation calculation.

        """
        result = 0.0
        for key in text_reference_counter:
            if key in self.__weights:
                result += text_reference_counter[key] * self.__weights[key][class_label]
        return result

    def __activation(self, value, fn):
        """Calculate the activation of a value.

        Args:
            value: A number input.
            fn:    A string identifier of the activation function to use.

        Returns:
            The result of the activation function.

        """
        if fn == "step":
            return self.__step_function(value)
        elif fn == "sigmoid":
            return self.__sigmoid_function(value)
        else:
            print("Invalid activation function provided: {}, falling back on to step activation.".format(fn))
            return self.__step_function(value)

    def __step_function(self, value):
        """Return 1 if value is greater than 0, -1 otherwise."""
        return 1 if value >= 0 else -1

    def __sigmoid_function(self, value):
        """Return the value of sigmoid transformation."""
        return 1 / (1 + exp(-value))
