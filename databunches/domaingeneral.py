from fastai import *
from fastai.text import *

import sentencepiece

class DomainGeneralDataBunch:
    """
    Creates and manages a DomainGeneralDataBunch.
    """

    def __init__(self, wikipedia_folder_path: Path, token_vocab_size: int, batch_size: int = 64, cpu_count: int = 8,
                 language_code: str = 'pl'):
        """
        Creates a DomainGeneralDataBunch, performing a sentencepiece-based tokenization on an entire dataset.
        :param wikipedia_folder_path: path to folder containing the previously downloaded Wikipedia contents
        :param token_vocab_size: size of the token vocabulary to use by sentencepiece
        :param batch_size: batch size to use during training
        :param cpu_count: number of CPUs to use during training
        :param language_code: code of language domain-general data is in (such as 'pl', 'eng', etc.)
        """
        dest = Path(wikipedia_folder_path/'docs')
        self.general_lm = (TextList.from_folder(dest, processor=[OpenFileProcessor(),SPProcessor(lang=language_code,
                           n_cpus=cpu_count, max_vocab_sz=token_vocab_size, include_bos=True, include_eos=True)])
                           .split_by_rand_pct(0.1, seed=42)
                           .label_for_lm()
                           .databunch(bs=batch_size, num_workers=cpu_count))

    def save(self, path):
        """
        Saves the DomainGeneralDataBunch to a given path.
        :param path: path where DomainGeneralDataBunch is to be saved
        :return:
        """
        self.general_lm.save(path)

    def get_databunch(self) -> DataBunch:
        """
        Returns a DataBunch object.
        :param path:
        :return: DataBunch
        """
        return self.general_lm
