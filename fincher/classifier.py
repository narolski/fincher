from fastai.text import *

import os

from fincher.consts import TARGET_CLASSIFIER_EXPORT_FILENAME


class SentimentClassificationLabel(Enum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


class SentimentClassificationSystemClassifier:
    """
    Performs document-level sentiment classification of a given input text document.
    """

    def __init__(self, models_path: Path, uncertainty_level: int):
        """
        Initializes the SentimentClassificationSystemClassifier
        :param models_path: path to trained models
        :param uncertainty_level: a maximum difference between the probability of document belonging to positive and
        negative class, at which the document is assigned a neutral label
        """
        self.classifier = load_learner(models_path, TARGET_CLASSIFIER_EXPORT_FILENAME)
        self.uncertainty_level = uncertainty_level

    def classify(self, document: str) -> (SentimentClassificationLabel, float, float):
        """
        Performs a document-level sentiment classification.
        :param document: input document in a string format
        :return: tuple (SentimentClassificationLabel, probability of belonging to a positive class, probability of
        belonging to a negative class)
        """
        prediction = self.classifier.predict(document)
        tensor_pos, tensor_neg = prediction[2][1] * 100, prediction[2][0] * 100
        prob_pos, prob_neg = tensor_pos.item(), tensor_neg.item()

        if abs(prob_pos - prob_neg) <= self.uncertainty_level:
            return SentimentClassificationLabel.NEUTRAL, prob_pos, prob_neg
        elif prob_pos > 50:
            return SentimentClassificationLabel.POSITIVE, prob_pos, prob_neg

        return SentimentClassificationLabel.NEGATIVE, prob_pos, prob_neg

    def format_return(self, path: Path, label: SentimentClassificationLabel, prob_pos: float, prob_neg: float):
        """
        Builds a string representation of classification result
        :param path: path to an input document
        :param label: SentimentClassificationLabel
        :param prob_pos: computed probability of document belonging to a positive class
        :param prob_neg: computed probability of document belonging to a negative class
        :return:
        """
        label_text = "positive"
        if label == SentimentClassificationLabel.NEGATIVE:
            label_text = "negative"
        elif label == SentimentClassificationLabel.NEUTRAL:
            label_text = "neutral"

        return "Document {} - Assigned label: {}, calculated probability of belonging to a positive class: {}, " \
               "calculated probability of " \
               "belonging to a negative class: {}".format(str(path), label, str(prob_pos), str(prob_neg))

    def classify_from_path(self, path: Path) -> str:
        """
        Performs a classification for a given document or directory contents (.txt files only).
        :param path: path to a file or directory
        :return:
        """
        if path.is_file():
            with open(str(path), mode='r') as file:
                document = file.read()
                classification, prob_pos, prob_neg = self.classify(document)
                return self.format_return(path, classification, prob_pos, prob_neg)

        elif path.is_dir():
            results = []
            docs = path.glob("**/*.txt")
            for doc_path in docs:
                with open(str(doc_path), mode='r') as file:
                    document = file.read()
                    classification, prob_pos, prob_neg = self.classify(document)
                    results.append(self.format_return(doc_path, classification, prob_pos, prob_neg))

            return "\n".join(results)
