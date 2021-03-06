from fastai import *
from fastai.text import *
import logging

class WikipediaDownloader:
    """
    WikipediaDownloader handles download process of an entire dump of a Wikipedia in a given language and allows for
    a downloaded dump to be processed into individual text documents, where each document is a single article entity.
    Based on a "Code-First Introduction to NLP" course (https://github.com/fastai/course-nlp).
    """
    @staticmethod
    def download(path: Path, language_code: str, min_article_len: int = 2600):
        """
        Downloads an entire dump of a Wikipedia for a given language_code and extracts it using wikiextractor.
        :param path: path in which a folder containing an extracted Wikipedia dump is stored
        :param language_code: language code for which a dump is to be downloaded from Wikipedia
        :param min_article_len: a minimum length of an article to be extracted from a dump by wikiextractor
        :return:
        """
        name = "{}wiki".format(language_code)
        if (path/name).exists():
            logging.info("{} has already been downloaded".format(name))
            return

        path.mkdir(exist_ok=True, parents=True)
        xml_file = "{}wiki-latest-pages-articles.xml".format(language_code)
        zip_file = "{}.bz2".format(xml_file)

        if not (path/xml_file).exists():
            logging.info("Downloading Wikipedia dump for language code {}...".format(language_code))
            download_url("https://dumps.wikimedia.org/{}/latest/{}".format(name, zip_file), path/zip_file,
                         show_progress=False)

            logging.info("Unzipping downloaded Wikipedia dump...")
            bunzip(path/zip_file)

        with working_directory(path):
            if not (path/'wikiextractor').exists():
                logging.info("Cloning wikiextractor from repo...")
                os.system('git clone https://github.com/attardi/wikiextractor.git')
                os.system('cd wikiextractor && git reset --hard 3162bb6c3c9ebd2d15be507aa11d6fa818a454ac && cd ..') # dirty fix

            logging.info("Running wikiextractor...")
            os.system("python wikiextractor/WikiExtractor.py --processes 4 --no_templates " +
            f"--min_text_length {min_article_len} --filter_disambig_pages --log_file log.txt -b 100G -q {xml_file}")

        shutil.move(str(path/'text/AA/wiki_00'), str(path/name))
        shutil.rmtree(path / 'text')

    @staticmethod
    def split_into_docs(path: Path, language_code: str) -> Path:
        """
        Splits into documents a downloaded XML of Wikipedia articles generated by wikiextractor.
        :param path: path in which a folder containing an extracted Wikipedia dump is stored
        :param language_code: language code for which has been downloaded from Wikipedia
        :return: path to a destination containing split text documents
        """
        destination = path/'docs'
        if destination.exists():
            logging.info("Documents have already been split to {}".format(destination))
            return destination

        destination.mkdir(exist_ok=True, parents=True)

        name = "{}wiki".format(language_code)
        title_re = re.compile(rf'<doc id="\d+" url="https://{language_code}.wikipedia.org/wiki\?curid=\d+" title="(['
                              rf'^"]+)">')
        lines = (path/name).open()
        file = None

        for i, line in enumerate(lines):
            if i % 100000 == 0:
                logging.info("Processed {} documents...".format(i))
            if line.startswith('<doc id="'):
                title = title_re.findall(line)[0].replace('/', '_')
                if len(title) > 150:
                    continue
                if file:
                    file.close()

                file = (destination/"{}.txt".format(title)).open('w')
            else:
                file.write(line)

        file.close()
        return destination