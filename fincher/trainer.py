from databunches.domaingeneral import DomainGeneralDataBunch
from databunches.domainspecific import DomainSpecificDataBunch
from databunches.targetclassifier import TargetClassifierDatabunch
from fincher.consts import (DOMAIN_GENERAL_DATABUNCH_NAME, DOMAIN_SPECIFIC_DATABUNCH_NAME,
                            TARGET_CLASSIFIER_DATABUNCH_NAME, GENERAL_LM_FILENAME, GENERAL_LM_VOCAB_FILENAME,
                            DOMAIN_SPECIFIC_ENCODER_FILENAME, DOMAIN_SPECIFIC_LM_FILENAME,
                            TARGET_CLASSIFIER_EXPORT_FILENAME)

from models.domaingenerallm import DomainGeneralLanguageModel
from models.domainspecificlm import DomainSpecificLanguageModel
from models.targetclassifier import TargetClassifier

from preprocessors.filmweb import FilmwebPreprocessor
from preprocessors.wikipedia import WikipediaDownloader
from fastai.text import *

import logging


class SentimentClassificationSystemTrainer:
    """
    Provides methods used for pre-processing training data and performing training of target classification models
    from pre-trained, domain-general language models fine-tuned into domain-specific language models.
    """
    @staticmethod
    def generate_databunches(wikipedia_path: Path, databunches_path: Path, language_code: str, vocab_size: int,
                             batch_size: int, num_cpus: int, filmwebplus_path: Path):
        """
        Generates databunches used for training document-level sentiment classification system.
        :param wikipedia_path: path where to store Wikipedia data
        :param databunches_path: path where to store generated DataBunch files
        :param language_code: language to use for downloading data from Wikipedia and tokenization ('pl', 'en', etc.)
        :param vocab_size: size of the token vocabulary to be used
        :param batch_size: batch size to be used in a training process
        :param num_cpus: number of CPUs to use when generating domain-general databunch and training sentencepiece model
        :param filmwebplus_path: path to filmwebplus.csv file containing the Filmweb+ dataset
        :return:
        """
        logging.info("Starting a generation of databunches...")
        wiki = WikipediaDownloader()
        wiki.download(wikipedia_path, language_code, min_article_len=2600)
        wiki.split_into_docs(wikipedia_path, language_code)

        databunches_path.mkdir(exist_ok=True, parents=True)

        logging.info("Generating a domain-general language model databunch. This may take a while...")
        domain_general = DomainGeneralDataBunch(wikipedia_path, vocab_size, batch_size, num_cpus, language_code)
        domain_general.save(Path(databunches_path / DOMAIN_GENERAL_DATABUNCH_NAME))

        logging.info("Processing filmwebplus.csv...")
        filmweb = FilmwebPreprocessor(filmwebplus_path)
        unlabelled_df = filmweb.get_entire_dataset()

        logging.info("Generating a domain-specific language model databunch...")
        domain_specific = DomainSpecificDataBunch(wikipedia_path, unlabelled_df, databunches_path)
        domain_specific.save(Path(databunches_path / DOMAIN_SPECIFIC_DATABUNCH_NAME))

        training, test = filmweb.get_labelled_train_test_split_dataset()
        labelled_df = pd.concat([training, test], sort=False)

        logging.info("Generating a target classifier databunch...")
        target = TargetClassifierDatabunch(wikipedia_path, databunches_path, labelled_df,
                                           domain_general.get_databunch(),
                                           batch_size=batch_size)
        target.save(Path(databunches_path / TARGET_CLASSIFIER_DATABUNCH_NAME))
        logging.info("Successfully generated all databunches.")

    @staticmethod
    def train(databunches_path: Path, batch_size: int, general_lm_epochs: int, models_path: Path,
              frozen_ds_lm_epochs: int, ds_lm_epochs: int, target_max_lr: float, target_training_epochs: tuple):
        """
        Trains a DomainGeneralLanguageModel, DomainSpecificLanguageModel and TargetClassifier
        :param databunches_path: path where to store generated DataBunch files
        :param batch_size: batch size to be used in a training process
        :param general_lm_epochs: epochs to train the entirely-unfrozen general language model for
        :param models_path: path where the model files are to be stored
        :param frozen_ds_lm_epochs: epochs to train the final layer of a domain-specific language model for
        :param ds_lm_epochs: epochs to train the entirely unfrozen domain-specific language model
        :param target_max_lr: maximum learning rate value to use when training a final classifier
        :param target_training_epochs: tuple of training epochs for a final classifier of format (1ep, 2ep, 3ep,
        4ep, 5ep)
        :return:
        """
        models_path.mkdir(exist_ok=True, parents=True)

        general_db = load_data(databunches_path, DOMAIN_GENERAL_DATABUNCH_NAME, bs=batch_size)
        general_lm = DomainGeneralLanguageModel(general_db)
        general_lm.train(batch_size=batch_size, training_epochs=general_lm_epochs)
        general_lm.save(Path(models_path / GENERAL_LM_FILENAME), Path(models_path / GENERAL_LM_VOCAB_FILENAME))

        specific_db = load_data(databunches_path, DOMAIN_SPECIFIC_DATABUNCH_NAME, bs=batch_size)
        specific_lm = DomainSpecificLanguageModel(specific_db, pretrained_model_path=Path(models_path /
                                                                                          GENERAL_LM_FILENAME),
                                                  pretrained_model_vocab_path=Path(models_path /
                                                                                   GENERAL_LM_VOCAB_FILENAME))
        specific_lm.train(batch_size=batch_size, frozen_training_epochs=frozen_ds_lm_epochs,
                          training_epochs=ds_lm_epochs)
        specific_lm.save(Path(models_path / DOMAIN_SPECIFIC_LM_FILENAME), Path(models_path /
                         DOMAIN_SPECIFIC_ENCODER_FILENAME))

        target_db = load_data(databunches_path, TARGET_CLASSIFIER_DATABUNCH_NAME, bs=batch_size)
        target_cl = TargetClassifier(target_db, Path(models_path / DOMAIN_SPECIFIC_ENCODER_FILENAME))
        target_cl.train(batch_size=batch_size, max_lr=target_max_lr, training_epochs=target_training_epochs)
        target_cl.export(Path(models_path / TARGET_CLASSIFIER_EXPORT_FILENAME))
