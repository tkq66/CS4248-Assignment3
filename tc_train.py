from porter import PorterStemmer
import re
from random import shuffle
from sys import argv


def stemTextFromFile(p, fileName):
    output = ""
    with open(fileName, "r") as infile:
        while True:
            word = ""
            line = infile.readline()
            if line == "":
                break
            for c in line:
                if c.isalpha():
                    word += c.lower()
                else:
                    if word:
                        output += p.stem(word, 0, len(word) - 1)
                        word = ""
                    output += c.lower()
    return output


def getStopWordSet(fileName):
    stopWordSet = set()
    with open(fileName, "r") as fp:
        stopWordSet = set(fp.read().rstrip().split("\n"))
    return stopWordSet


def getTrainingClassList(fileName, tcLocation="/home/course/cs4248/"):
    defaultLocation = "/home/course/cs4248/"
    isReplaceFileLoc = (tcLocation != defaultLocation)
    trainingClassList = []
    with open(fileName, "r") as fp:
        for line in fp:
            preprocessedLine = line.strip()
            if preprocessedLine == "":
                continue
            preprocessedLine = re.sub(defaultLocation, tcLocation, preprocessedLine) if isReplaceFileLoc else preprocessedLine
            trainingClassList.append(tuple(preprocessedLine.split()))
    shuffle(trainingClassList)
    return trainingClassList


def main():
    stopWordFileName = argv[1]
    trainClassListFileName = argv[2]
    outputModelFileName = argv[3]
    p = PorterStemmer()
    stopWordSet = getStopWordSet(stopWordFileName)
    trainingClassList = getTrainingClassList(trainClassListFileName, tcLocation="")
    print(trainingClassList)


if __name__ == "__main__":
    main()
