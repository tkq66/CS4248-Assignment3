from porter import PorterStemmer
import re
from sys import argv

totalClasses = 0
initialWeightValue = 0
inititialBiasValue = 1
biasWeightKey = "__bias__"
weights = {}


def getTextVectorFromFile(fileName):
    # Read the text from file, remove all the spaces, and split into a vector
    textVector = []
    with open(fileName, "r") as fp:
        textVector = fp.read().split()
    return textVector


def getStopWordSet(fileName):
    stopWordSet = set()
    with open(fileName, "r") as fp:
        stopWordSet = set(fp.read().strip().split("\n"))
    return stopWordSet


def getTrainingClassReference(fileName, tcLocation="/home/course/cs4248/"):
    defaultLocation = "/home/course/cs4248/"
    isReplaceFileLoc = (tcLocation != defaultLocation)
    trainingClassReference = {}
    with open(fileName, "r") as fp:
        for line in fp:
            preprocessedLine = line.strip()
            if preprocessedLine == "":
                continue
            preprocessedLine = re.sub(defaultLocation, tcLocation, preprocessedLine) if isReplaceFileLoc else preprocessedLine
            filePath, className = preprocessedLine.split()
            if className not in trainingClassReference:
                trainingClassReference[className] = []
            trainingClassReference[className].append(tuple((filePath, className)))
    return trainingClassReference


def preprocessWord(rawInput, p, stopWordSet):
    # Normalize the text to lowercase
    inputWord = rawInput.lower() if rawInput.isalpha() else rawInput
    # Ignore stopwords
    if inputWord in stopWordSet:
        return None
    # Stem the word if it is indeed a word
    word = p.stem(inputWord, 0, len(inputWord) - 1) if rawInput.isalpha() else inputWord
    return word


def preprocessText(textVector, p, stopWordSet):
    global weights
    # Initialize the counter reference for each word, with an auxiliary bias identity modifier
    textReferenceCounter = {biasWeightKey: 1}
    # Increment word counts and encode the word vector into numerical values
    totalWordCount = 0
    for i in range(len(textVector)):
        word = preprocessWord(textVector[i], p, stopWordSet)
        if word is None:
            continue
        # Initialize weight for word not yet seen
        if word not in weights:
            weights[word] = [initialWeightValue] * totalClasses
        # Increment total word counts
        textReferenceCounter[word] = textReferenceCounter.get(word, 0) + 1
        totalWordCount += 1
    # Normalize the word frequency so that texts of different length will
    # contribute to the machine in the same scale
    for word in textReferenceCounter:
        if word == biasWeightKey:
            continue
        textReferenceCounter /= totalWordCount
    return textReferenceCounter


def activation(value, fn="step"):
    if fn == "step":
        return 1 if value > 0 else -1
    else:
        raise ValueError("Please provide a valid activation function type.")


def affineTransformation(weightIndex, textReferenceCounter):
    result = 0.0
    for key in textReferenceCounter:
        result = textReferenceCounter[key] * weights[key][weightIndex]
    return result


def forwardPass(weightIndex, textVector, p, stopWordSet):
    textReferenceCounter = preprocessText(textVector, p, stopWordSet)
    affineResult = affineTransformation(weightIndex, textReferenceCounter)
    result = activation(affineResult, fn="step")
    return result, textReferenceCounter


def updateWeights(textReferenceCounter, lr, networkOutput):
    global weights
    for word in textReferenceCounter:
        # Assuming that the classes are one-hot encoded, the target output is 1
        delta = lr * (1 - networkOutput) * textReferenceCounter[word]
        weights[word] += delta


def train(p, stopWordSet, trainingClassReference, lr=0.01):
    assert lr > 0
    for index, className in trainingClassReference:
        for trainingClass in trainingClassReference[className]:
            filePath, classLabel = trainingClass
            textVector = getTextVectorFromFile(filePath)
            # Send new input through forward pass
            result, textReferenceCounter = forwardPass(index, textVector, p, stopWordSet, lr)
            # Update the weights
            updateWeights(textReferenceCounter, lr, result)


def main():
    stopWordFileName = argv[1]
    trainClassListFileName = argv[2]
    outputModelFileName = argv[3]

    trainingClassReference = getTrainingClassReference(trainClassListFileName, tcLocation="")
    global totalClasses
    totalClasses = trainingClassReference.keys()
    global weights
    weights = {biasWeightKey: [inititialBiasValue] * len(totalClasses)}
    p = PorterStemmer()
    stopWordSet = getStopWordSet(stopWordFileName)

    train(p, stopWordSet, trainingClassReference)


if __name__ == "__main__":
    main()
