# Fincher

A document-level sentiment classification system written for the purpose of classifying Polish movie reviews based on
 their underlying sentiment.
 
 ## Pre-requisites
 
 Install requirements using `pipenv` by activating the environment and typing `pipenv install` command.
 
 In order to perform the classification, an underlying target classifier model needs to be trained. In order to do so
 , download the contents of the Filmweb+ dataset:
 ```bash
./download.sh
```
then run:

```bash
python main.py prepare_data
```
and finally:
```bash
python main.py train
```

## Classification

In order to perform a classification run:
```bash
python main.py classify [path_to_doc]
```
where `[path_to_doc]` is a path to a document or a directory containing multiple .txt files to be classified.

## Parameters

For available parameters, type:
```bash
python main.py [method] --help
```
where a provision of `[method]` lists all available parameters for a given system mode.