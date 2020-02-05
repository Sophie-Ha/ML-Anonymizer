# ML-Anonymizer
Code for my Bachelor thesis on textual anonymization using ML.
With textual anonymization, the main challenge lies in identifying which words make up personally identifying information and, thus, need to be replaced by anon tokens or pseudonyms.
This repository contains the implementations of two different ML-approaches for identifying personal data and different evaluation scripts.

## Pointer Network
The base idea of this approach is to process an input sentence with a pointer network [[1]](#1) to generate pointers that mark the words in the input sentence that should be anonymized.

Usage examples can be found in 
- `0511_experiments.py`
- `0606_experiments.py`
- `misclassification_script.py`
- `ptr_timer.py`

## Binary Classification
First, the words of an input sentence are embedded.
A binary classification model, then, predicts for each word, whether to anonymize it or not.

Usage examples can be found in
- `0511_experiments.py`
- `0606_experiments.py`
- `misclassification_script.py`
- `rf_analysis.py`
- `sentiment.py`

## Datasets

### CoNLL2003
This dataset [[2]](#2) was originally used for Named Entity Recognition.
The task is, however, quite similar to recognizing personal data.
As a result, a modified version of this dataset is used for training and evaluation.
The data can be downloaded [from this repo](https://github.com/Franck-Dernoncourt/NeuroNER/tree/master/neuroner/data/conll2003/en).
The script `data/process_conll.py` is used to read the downloaded files and transform them into the custom data format used by my model setup.

### ITAC
This dataset consists of emails where personal data has been tagged. [[3]](#3)

### GloVe embeddings
For usage of GloVe [[4]](#4) embeddings, the corresponding embedding files can be downloaded [here](https://nlp.stanford.edu/projects/glove/).

### Word frequencies
To compute additional sentence embeddings from word embeddings, a list of word frequencies is needed.
The corresponding data can be downloaded [here](https://github.com/IlyaSemenov/wikipedia-word-frequency/blob/master/results/enwiki-20190320-words-frequency.txt).

### Sentiment analysis
For sentiment analysis, more specifically polarity and subjectivity classification a set of movie reviews [[5]](#5) is used.
The data is available [here](http://www.cs.cornell.edu/people/pabo/movie-review-data/).

### Keyphrase extraction
To test keyphrase extraction, a dataset consisting of news articles [[6]](#6) is used.

## References
<a id="1">[1]</a>
Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly.
“Pointer networks.”
In: Advances in Neural Information Processing Systems. 2015, pp. 2692–2700.

<a id="2">[2]</a>
Erik F Tjong Kim Sang and Fien De Meulder.
“Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition.”
In: Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4. Association for Computational Linguistics.
2003, pp. 142–147.

<a id="3">[3]</a>
Ben Medlock.
“An Introduction to NLP-based Textual Anonymisation.”
In: LREC. Citeseer. 2006, pp. 1051–1056.

<a id="4">[4]</a>
Jeffrey Pennington, Richard Socher, and Christopher Manning.
“Glove: Global vectors for word representation.”
In: Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014, pp. 1532–1543.

<a id="5">[5]</a>
Bo Pang and Lillian Lee.
“A sentimental education: Sentiment analysis using subjectivity summarization based on minimum cuts.”
In: Proceedings of the 42nd annual meeting on Association for Computational Linguistics.
Association for Computational Linguistics. 2004, p. 271.

<a id="6">[6]</a>
Luís Marujo, Anatole Gershman, Jaime Carbonell, Robert Frederking, and João P Neto.
“Supervised topical key phrase extraction of news stories using crowdsourcing, light filtering and co-reference normalization.”
In: arXiv preprint arXiv:1306.4886 (2013).