from fastai.text import *
from fastai import *
from fastai.basic_data import DataBunch

class DomainGeneralLanguageModel:
    """
    Handles creation and pre-training of a domain-general language model.
    """
    def __init__(self, lm_databunch : DataBunch, drop_mult: float = 0.1, wd: float = 0.1):
        """
        Initializes the DomainGeneralLanguageModel
        :param lm_databunch: DomainGeneralLanguageModelDataBunch to use for training.
        :param drop_mult: dropout multiplication amount to use in AWD-LSTM
        :param wd: weight dropout to use in AWD-LSTM
        """
        self.learn = language_model_learner(lm_databunch, AWD_LSTM, drop_mult=drop_mult, wd=wd, pretrained=False).to_fp16()

    def train(self, batch_size : int = 64, moms: slice = (0.8, 0.7), training_epochs: int = 10):
        """
        Trains the DomainGeneralLanguageModel.
        :param batch_size: size of batch of training data to use during training
        :param moms: slice of values of momentums to use during training (from mom_min to mom_max)
        :param training_epochs: training epochs to use in a training instance
        :return:
        """
        lr = 1e-3
        lr *= batch_size / (48 / (128 / batch_size))

        self.learn.unfreeze()
        self.learn.fit_one_cycle(training_epochs, lr, moms)

    def save(self, model_path: Path, vocab_path: Path):
        """
        Saves the DomainGeneralLanguageModel.
        :param model_path: path to store the weights of a model
        :param vocab_path: path to store the vocabulary of a model
        :return:
        """
        self.learn.to_fp32().save(model_path, with_opt=False)
        self.learn.data.vocab.save(vocab_path.with_suffix('.pkl'))

    def get_model(self) -> LanguageLearner:
        """
        Returns a current state of the LanguageLearner instance.
        :return: LanguageLearner object
        """
        return self.learn
