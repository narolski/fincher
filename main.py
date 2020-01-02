import argparse
import logging

from fincher.classifier import SentimentClassificationSystemClassifier
from fincher.trainer import SentimentClassificationSystemTrainer
from fastai.text import *

import torch


def main():
    # torch.cuda.set_device(0)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog="Document Sentiment Classification System")
    subparsers = parser.add_subparsers(title="Available modes")

    classify_parser = subparsers.add_parser('classify', help='performs a classification of a document or multiple '
                                                             'documents in a given path')
    classify_parser.add_argument('path', type=str, help='path to .txt file or folder containing multiple '
                                                                  '.txt '
                                                                 'files only')
    classify_parser.add_argument('--models_path', type=str, help='path to store trained model files',
                              default="~/data/models/")
    classify_parser.add_argument('--uncertainty_level', type=int, help='an maximum amount of difference between the '
                                                                       'computed probability of a document belonging '
                                                                       'to a positive class and the probability of '
                                                                       'belonging to a negative class at which system '
                                                                       'assigns the document the NEUTRAL label',
                                 default=5)

    data_parser = subparsers.add_parser('prepare_data', help='prepares databunches used for training domain-general '
                                                              'language model, domain-specific language model and '
                                                              'target classifier model')
    data_parser.add_argument('--wikipedia_path', type=str, help='path to store contents of Wikipedia',
                             default="~/data/wikipedia/")
    data_parser.add_argument('--filmwebplus_path', type=str, help='path to filmwebplus.csv file',
                              default="~/data/filmwebplus/filmwebplus.csv")
    data_parser.add_argument('--language_code', type=str, help='code of language to use when downloading data from '
                                                                'Wikipedia and for sentencepiece-based tokenization',
                              default="pl")
    data_parser.add_argument('--min_article_len', type=int, help='minimum length of Wikipedia article in chars',
                              default=2600)
    data_parser.add_argument('--batch_size', type=int, help='batch size to use during latter training',
                              default=64)
    data_parser.add_argument('--vocab_size', type=int, help='token vocabulary size to use',
                              default=32000)
    data_parser.add_argument('--cpu_count', type=int, help='number of CPUs to use during databunch generation',
                              default=8)
    data_parser.add_argument('--databunches_path', type=str, help='path to store databunch files',
                              default="~/data/databunches/")

    train_parser = subparsers.add_parser('train', help='trains a domain-general language model, a domain-specific '
                                                       'language model and target classifier model used for '
                                                       'predictions')
    train_parser.add_argument('--databunches_path', type=str, help='path to stored databunch files',
                             default="~/data/databunches/")
    train_parser.add_argument('--models_path', type=str, help='path to store trained model files',
                              default="~/data/models/")
    train_parser.add_argument('--batch_size', type=int, help='batch size to use during latter training',
                              default=64)
    train_parser.add_argument('--general_lm_epochs', type=int, help='# of domain-general language model '
                                                                    'pre-training epochs', default=10)
    train_parser.add_argument('--ds_lm_frozen_epochs', type=int, help='# of domain-specific language model '
                                                                    'fine-tuning epochs in a first step of training '
                                                                      'performed on a last model layer only',
                              default=2)
    train_parser.add_argument('--ds_lm_epochs', type=int, help='# of domain-specific language model '
                                                                      'fine-tuning epochs in a second step of '
                                                               'training performed on all model layers',
                              default=2)
    train_parser.add_argument('--target_max_lr', type=float, help='maximum learning rate to use when training target '
                                                                 'classifier',
                              default=float(2e-2))
    train_parser.add_argument('--target_step1_epochs', type=int, help='# of epochs in the first step of target '
                                                                        'classifier training',
                              default=2)
    train_parser.add_argument('--target_step2_epochs', type=int, help='# of epochs in the second step of target '
                                                                      'classifier training',
                              default=2)
    train_parser.add_argument('--target_step3_epochs', type=int, help='# of epochs in the third step of target '
                                                                      'classifier training',
                              default=2)
    train_parser.add_argument('--target_step4_epochs', type=int, help='# of epochs in the fourth step of target '
                                                                      'classifier training',
                              default=1)
    train_parser.add_argument('--target_step5_epochs', type=int, help='# of epochs in the fifth step of target '
                                                                      'classifier training',
                              default=1)

    args = parser.parse_args()

    if args.__contains__('cpu_count'):
        trainer = SentimentClassificationSystemTrainer()
        trainer.generate_databunches(wikipedia_path=Path(args.wikipedia_path).expanduser(), filmwebplus_path=Path(
            args.filmwebplus_path).expanduser(),
                             language_code=args.language_code, vocab_size=args.vocab_size,
                             batch_size=args.batch_size, num_cpus=args.cpu_count, databunches_path=Path(
                args.databunches_path).expanduser())

    elif args.__contains__('general_lm_epochs'):
        trainer = SentimentClassificationSystemTrainer()
        trainer.train(databunches_path=Path(args.databunches_path).expanduser(), batch_size=args.batch_size,
              general_lm_epochs=args.general_lm_epochs, models_path=Path(args.models_path).expanduser(),
              frozen_ds_lm_epochs=args.ds_lm_frozen_epochs, target_max_lr=args.target_max_lr,
              target_training_epochs=(args.target_step1_epochs, args.target_step2_epochs, args.target_step3_epochs,
                                      args.target_step4_epochs, args.target_step5_epochs), ds_lm_epochs=args.ds_lm_epochs)

    elif args.__contains__('classify_path'):
        classifier = SentimentClassificationSystemClassifier(models_path=args.models_path, uncertainty_level=args.uncertainty_level)
        result = classifier.classify_from_path(args.path)
        print(result)

    else:
        parser.print_usage()



if __name__ == "__main__":
    main()
