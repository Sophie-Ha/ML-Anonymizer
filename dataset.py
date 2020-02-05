import copy
import numpy as np
import re

from nltk import sent_tokenize


class Document:

    def __init__(self, sentences=None, annotated=None, predicted=None, raw_text=None):
        """
        :param list[list[str]]|None sentences: list of word lists per sentence
        :param list[list[int]]|None annotated: list of annotations per sentence (0: no annotation, 1: beginning of sensitive token, 2: inner word of sensitive token)
        """
        self.sentences = sentences if sentences else []
        self.annotated = annotated if annotated else []
        self.predicted = predicted if predicted else []
        self.raw_text = raw_text
        self._word_count = np.sum([len(s) for s in self.sentences])
        self._annotation_count = np.sum([np.sum(a) for a in self.annotated])
        self._sentence_count = len(self.sentences)

    @property
    def word_count(self):
        self._word_count = np.sum([len(s) for s in self.sentences])
        return self._word_count

    @property
    def annotation_count(self):
        self._annotation_count = np.sum([np.sum(a) for a in self.annotated])
        return self._annotation_count

    @property
    def sentence_count(self):
        self._sentence_count = len(self.sentences)
        return self._sentence_count

    def to_string_standardized(self):
        """Creates standardized string representation of the document"""
        s = ""
        for sent, anno in zip(self.sentences, self.annotated):
            for w, i in zip(sent, anno):
                s += w + " " + str(i) + "\n"
            s += "\n"
        return s

    def create_from_text(self):
        """Loads raw text and converts it into list[list[str]]"""
        punctuation = re.compile("[,\.;:'\?!\"\(\)\[\]]")
        # try:
        for sent in sent_tokenize(self.raw_text):
            matches = re.findall(punctuation, sent)
            for m in matches:
                sent = re.sub(re.escape(m), " " + m + " ", sent)
            words = sent.split()
            anno = [0] * len(words)
            self.sentences.append(list(words))
            self.annotated.append(list(anno))
        # except:
        #    print(self.raw_text)

    def create_text(self, anon=True):
        output_str = ""
        for sent, anno in zip(self.sentences, self.annotated):
            if anon:
                w_list = []
                for w, a in zip(sent, anno):
                    if a != 0:
                        w_list.append("<ANON>")
                    else:
                        w_list.append(w)
                output_str += " ".join(w_list) + "\n"
            else:
                output_str += " ".join(sent)
        return output_str.strip()


class DataSet:

    def __init__(self, path, reduce_to_one=True):
        self.path = path
        self.documents = []
        self._word_count = 0
        self._annotation_count = 0
        self._sentence_count = 0
        self.red_to_one = reduce_to_one

    @property
    def word_count(self):
        self._word_count = np.sum([d.word_count for d in self.documents])
        return self._word_count

    @property
    def annotation_count(self):
        self._annotation_count = np.sum([d.annotation_count for d in self.documents])
        return self._annotation_count

    @property
    def sentence_count(self):
        self._sentence_count = np.sum([d.sentence_count for d in self.documents])
        return self._sentence_count

    def get_merged_annotations(self):
        """creates one list of all sentence annotation lists"""
        res = []
        for doc in self.documents:
            res += doc.annotated
        return res

    def split_into_multiple(self, n, base_path):
        doc_size = int(len(self.documents)/n)
        for i in range(n-1):
            ds = DataSet(base_path + str(i) + ".txt")
            ds.add_documents(self.documents[i*doc_size:(i+1)*doc_size])
            ds.write_data()
        ds = DataSet(base_path + str(n-1) + ".txt")
        ds.add_documents(self.documents[(n-1)*doc_size:])
        ds.write_data()

    def read_data(self):
        with open(self.path, encoding="utf8") as f:
            cur_document = Document()
            cur_sentence = []
            cur_anno = []

            for line in f.readlines():
                if len(line) > 1:
                    temp_word_list = line.split()
                    if temp_word_list[0] == "-DOCSTART-":
                        if cur_document.sentences:
                            self.add_document(cur_document)
                            cur_document = Document()
                            cur_sentence = []
                            cur_anno = []
                    else:
                        cur_sentence.append(temp_word_list[0])
                        if self.red_to_one and int(temp_word_list[1]) > 0:
                            cur_anno.append(1)
                        else:
                            cur_anno.append(int(temp_word_list[1]))
                elif cur_sentence:
                    cur_document.sentences.append(list(cur_sentence))
                    cur_document.annotated.append(list(cur_anno))
                    cur_sentence = []
                    cur_anno = []

            if cur_document.sentences:
                self.add_document(cur_document)

    def write_data(self):
        with open(self.path, "w+", encoding="utf8") as f:
            for doc in self.documents:
                f.write("-DOCSTART-\n")
                f.write(doc.to_string_standardized())

    def read_multiple(self, paths):
        """Reads multiple datasets and combines them into one"""
        for path in paths:
            with open(path, encoding="utf8") as f:
                cur_document = Document()
                cur_sentence = []
                cur_anno = []

                for line in f.readlines():
                    if len(line) > 1:
                        temp_word_list = line.split()
                        if temp_word_list[0] == "-DOCSTART-":
                            if cur_document.sentences:
                                self.add_document(cur_document)
                                cur_document = Document()
                                cur_sentence = []
                                cur_anno = []
                        else:
                            if len(temp_word_list) <= 1:
                                print(line)
                            cur_sentence.append(temp_word_list[0])
                            if self.red_to_one and int(temp_word_list[1]) > 0:
                                cur_anno.append(1)
                            else:
                                cur_anno.append(int(temp_word_list[1]))
                    elif cur_sentence:
                        cur_document.sentences.append(list(cur_sentence))
                        cur_document.annotated.append(list(cur_anno))
                        cur_sentence = []
                        cur_anno = []

                if cur_document.sentences:
                    self.add_document(cur_document)

    def add_document(self, document):
        self.documents.append(copy.copy(document))

    def add_documents(self, documents):
        for document in documents:
            self.documents.append(copy.copy(document))
