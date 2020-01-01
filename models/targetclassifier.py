from fastai.text import *
from fastai import *
from fastai.basic_data import DataBunch


class TargetClassifier:
    """
    Handles creation and training of a target classifier.
    """
    def __init__(self, class_databunch: DataBunch, encoder_path: Path, drop_mult: float = 1.0, wd: float = 0.1):
        """
        Initializes the TargetClassifier
        :param class_databunch: TargetClassifierDataBunch to use for training.
        :param drop_mult: dropout multiplication amount to use in AWD-LSTM
        :param wd: weight dropout to use in AWD-LSTM
        :param encoder_path: path to a fine-tuned language moder encoder
        """
        self.learn = text_classifier_learner(class_databunch, AWD_LSTM, drop_mult=drop_mult, pretrained=False,
                                             wd=wd).to_fp16()
        self.learn.load_encoder(str(encoder_path))

    def train(self, batch_size: int = 64, moms: slice = (0.8, 0.7), training_epochs: tuple = (2, 2, 2, 1, 1),
              max_lr: float = 2e-2):
        """
        Trains the TargetClassifier.
        :param max_lr: maximum learning rate value to use during training
        :param batch_size: size of batch of training data to use during training
        :param moms: slice of values of momentums to use during training (from mom_min to mom_max)
        :param training_epochs: slice of training epochs to use for training instances, where first is applied to a
        training of a model entirely unfrozen except the final layer, second - for the training of a model unfrozen to
        the 2-nd layer from the end, third - for the model unfrozen to the 3rd layer of a model from the end,
        fourth and fifth - to an entirely unfrozen model training
        :return:
        """
        if len(training_epochs) is not 5:
            raise Exception("Invalid amount of training epochs provided")

        self.learn.freeze()
        lr = max_lr * batch_size / (48 / (128 / batch_size))

        first_ep, second_ep, third_ep, fourth_ep, fifth_ep = training_epochs

        self.learn.fit_one_cycle(first_ep, lr, moms)

        self.learn.freeze_to(-2)
        self.learn.fit_one_cycle(second_ep, slice(lr / (2.6 ** 4), lr), moms=(0.8, 0.7))

        self.learn.freeze_to(-3)
        self.learn.fit_one_cycle(third_ep, slice(lr / 2 / (2.6 ** 4), lr / 2), moms=(0.8, 0.7))

        self.learn.unfreeze()
        self.learn.fit_one_cycle(fourth_ep, slice(lr / 10 / (2.6 ** 4), lr / 10), moms=(0.8, 0.7))
        self.learn.fit_one_cycle(fifth_ep, slice(lr / 10 / (2.6 ** 4), lr / 10), moms=(0.8, 0.7))

    def export(self, model_path: Path):
        """
        Exports the TargetClassifier.
        :param model_path: path to store the exported classifier
        :return:
        """
        self.learn.save(model_path, with_opt=False)

    def get_model(self) -> RNNLearner:
        """
        Returns a current state of the RNNLearner instance.
        :return: RNNLearner object
        """
        return self.learn
