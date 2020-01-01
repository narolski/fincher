from fastai.text import *
import torch

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
        prediction = self.classifier.predict(document)
        tensor_pos, tensor_neg = prediction[2][1]*100, prediction[2][0]*100
        prob_pos, prob_neg = tensor_pos.item(), tensor_neg.item()

        if abs(prob_pos - prob_neg) <= self.uncertainty_level:
            return SentimentClassificationLabel.NEUTRAL, prob_pos, prob_neg
        elif prob_pos > 50:
            return SentimentClassificationLabel.POSITIVE, prob_pos, prob_neg

        return SentimentClassificationLabel.NEGATIVE, prob_pos, prob_neg








