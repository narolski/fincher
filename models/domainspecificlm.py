from fastai.text import *
from fastai import *
from fastai.basic_data import DataBunch

class DomainSpecificLanguageModel:
    """
    Handles creation and fine-tuning of a domain-specific language model.
    """
    def __init__(self, lm_databunch : DataBunch, pretrained_model_path: Path, pretrained_model_vocab_path: Path,
                 drop_mult: float = 1.0, wd: float = 0.1,):
        """
        Initializes the DomainSpecificLanguageModel
        :param lm_databunch: DomainSpecificLanguageModelDataBunch to use for training.
        :param drop_mult: dropout multiplication amount to use in AWD-LSTM
        :param wd: weight dropout to use in AWD-LSTM
        :param pretrained_model_path: path to a pre-trained language model vocab file
        :param pretrained_model_vocab_path: path to a pre-trained language model file
        """
        pretrained_fnames = (str(pretrained_model_path), str(pretrained_model_vocab_path))
        self.learn = language_model_learner(lm_databunch, AWD_LSTM, drop_mult=drop_mult, wd=wd, pretrained=False,
                                            pretrained_fnames=pretrained_fnames)

    def train(self, batch_size: int = 64, moms: slice = (0.8, 0.7), frozen_training_epochs: int = 2, training_epochs:
        int = 10):
        """
        Trains the DomainSpecificLanguageModel.
        :param batch_size: size of batch of training data to use during training
        :param moms: slice of values of momentums to use during training (from mom_min to mom_max)
        :param frozen_training_epochs: training epochs to use in a training instance performed on a model frozen to 
        all layers except for a final one
        :param training_epochs: training epochs to use in a training instance on an entirely unfrozen model
        :return:
        """
        lr = 1e-3
        lr *= batch_size / (48 / (128 / batch_size))

        self.learn.fit_one_cycle(frozen_training_epochs, lr, moms)

        self.learn.unfreeze()
        self.learn.fit_one_cycle(training_epochs, lr, moms)

    def save(self, model_path: Path, encoder_path: Path):
        """
        Saves the DomainSpecificLanguageModel.
        :param model_path: path to store the weights of a model
        :param encoder_path: path to store the model's encoder
        :return:
        """
        self.learn.save(model_path, with_opt=False)
        self.learn.save_encoder(str(encoder_path))

    def get_model(self) -> LanguageLearner:
        """
        Returns a current state of the LanguageLearner instance.
        :return: LanguageLearner object
        """
        return self.learn
