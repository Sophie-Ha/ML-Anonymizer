from dataset import DataSet, Document
import numpy as np


class EmbeddingProcessor:

    def __init__(self, embedding_man, wiki_path):
        self.embedding_man = embedding_man
        self.wiki_freq_dict = None
        self.total_wiki_dict_count = 0
        self.wiki_path = wiki_path

    def get_sent_emb_simple(self, doc):
        """
        Creates sentence embeddings according to arora2016
        :param doc: Document object
        :param wiki_path: path to file containing word frequencies form wikipedia
        :return: list of sentence embedding vectors
        """

        if self.wiki_freq_dict is None:
            self._load_in_wiki_dic(self.wiki_path)

        sent_vecs = np.zeros((self.embedding_man.dim, len(doc.sentences)))

        for i in range(len(doc.sentences)):
            res = self._get_simple_sentence_vec(doc.sentences[i])
            sent_vecs[:, i] = res.reshape(self.embedding_man.dim)

        u, _, _ = np.linalg.svd(sent_vecs)
        svd_mat = np.matmul(u[:, 0].reshape((self.embedding_man.dim, 1)),
                            u[:, 0].reshape((self.embedding_man.dim, 1)).transpose())

        sentence_embeddings = []
        for i in range(len(doc.sentences)):
            vec = sent_vecs[:, i].reshape((self.embedding_man.dim, 1)) - \
                  np.matmul(svd_mat,sent_vecs[:, i].reshape((self.embedding_man.dim, 1)))
            sentence_embeddings.append(vec)

        return sentence_embeddings

    def _get_simple_sentence_vec(self, sentence, a=0.000000000001):
        """
        creates weighted sentence vector needed for the sentence embedding from "tough to beat baseline"
        :param sentence: word list from one sentence
        :param a: factor for word weights
        :return: preliminary embedding vector for the given sentence
        """
        vec = np.zeros((self.embedding_man.dim, 1))
        for w in sentence:
            temp_vec = np.reshape(self.embedding_man.get_embedding_vec(w), (self.embedding_man.dim, 1))
            if w in self.wiki_freq_dict:
                vec = vec + 100 * temp_vec * a / (a + self.wiki_freq_dict[w])
            else:
                vec = vec + 100 * temp_vec * a / (a + 1)
        return vec / len(sentence)

    def _load_in_wiki_dic(self, path):
        """
        requires https://raw.githubusercontent.com/IlyaSemenov/wikipedia-word-frequency/master/results/enwiki-20190320-words-frequency.txt
        loads in a dictionary with relative word frequencies from wikipedia
        :param path: path to the file containing the word frequencies
        :return:
        """
        with open(path, encoding="utf8") as f:
            wiki_dict = {}
            total_count = 0
            for line in f.readlines():
                word = line.split()[0]
                count = int(line.split()[1])
                total_count += count
                wiki_dict[word] = count
        self.wiki_freq_dict = wiki_dict
        self.total_wiki_dict_count = total_count

