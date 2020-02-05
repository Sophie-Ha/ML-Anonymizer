import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import umap

from embeddings.embedding_manager import EmbeddingMan, GloveEmbeddings, BPEmbeddings, CombinedEmbeddings
from dataset import Document, DataSet
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, Embeddings
from flair.data import Sentence


class DataEmbedder:

    def __init__(self, create_sent_emb=False, wiki_path=""):
        self.create_sent_emb = create_sent_emb
        self.wiki_freq_dict = None
        self.total_wiki_dict_count = 0
        self.wiki_path = wiki_path
        self.embedding_dim = 0

    def reduce_dataset(self, ds, path=""):
        embedded, word_list, annotation_list = self.embed_dataset(ds)
        print("Embedded dataset. Performing Dimensionality Reduction")
        reducer = umap.UMAP()
        reduced = reducer.fit_transform(embedded)

        res_df = pd.DataFrame(reduced)
        res_df["word"] = word_list
        res_df["annotated"] = annotation_list
        if path:
            res_df.to_csv(path)
        return res_df

    def plot_reduced(self, df=None, path=None, save=None):
        assert df is not None or path is not None, "Needs a dataframe or path to a stored dataframe to visualize"
        if df is None:
            df = pd.read_csv(path, index_col=0)
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df["annotated"])
        if save is not None:
            plt.savefig(save)
        plt.show()

    def embed_dataset(self, ds):
        embedding_list = []
        word_list = []
        annotation_list = []

        for doc in ds.documents:
            doc_embeddings = self._embed_document(doc)
            for w_list, a_list, emb_list in zip(doc.sentences, doc.annotated, doc_embeddings):
                for w, a, e in zip(w_list, a_list, emb_list):
                    word_list.append(w)
                    annotation_list.append(a)
                    embedding_list.append(e)

        embedding_matrix = np.reshape(np.array(embedding_list), (len(word_list), -1))
        print("Embedding matrix has shape {}".format(embedding_matrix.shape))
        return embedding_matrix, word_list, annotation_list

    def _embed_sentence(self, sent):
        raise NotImplementedError

    def _embed_document(self, doc):
        doc_embeddings = []
        for sent in doc.sentences:
            doc_embeddings.append(self._embed_sentence(sent))
        if not self.create_sent_emb:
            return doc_embeddings
        simple_sent_embeddings = self.get_simple_sentence_vecs(doc_embeddings, doc)
        new_doc_embeddings = []
        for sent_list, sent_emb in zip(doc_embeddings, simple_sent_embeddings):
            new_sent_list = []
            for w in sent_list:
                new_w = np.concatenate((np.reshape(w, (self.embedding_dim, 1)),
                                        np.reshape(sent_emb, (self.embedding_dim, 1))))
                new_sent_list.append(new_w)
            new_doc_embeddings.append(new_sent_list)
        return new_doc_embeddings

    def get_simple_sentence_vecs(self, doc_embeddings, doc):
        """
        :param list[list[numpy.array]] doc_embeddings:
        :param Document doc:
        :return:
        """

        if self.wiki_freq_dict is None:
            self._load_in_wiki_dic(self.wiki_path)

        sent_vecs = np.zeros((self.embedding_dim, len(doc.sentences)))

        for i in range(len(doc.sentences)):
            res = self._get_temp_sentence_vec(doc_embeddings[i], doc.sentences[i])
            sent_vecs[:, i] = res.reshape(self.embedding_dim)

        u, _, _ = np.linalg.svd(sent_vecs)
        svd_mat = np.matmul(u[:, 0].reshape((self.embedding_dim, 1)),
                            u[:, 0].reshape((self.embedding_dim, 1)).transpose())

        sentence_embeddings = []
        for i in range(len(doc.sentences)):
            vec = sent_vecs[:, i].reshape((self.embedding_dim, 1)) - \
                  np.matmul(svd_mat, sent_vecs[:, i].reshape((self.embedding_dim, 1)))
            sentence_embeddings.append(vec)

        return sentence_embeddings

    def _get_temp_sentence_vec(self, sentence, word_list, a=0.000000000001):
        """
        creates weighted sentence vector needed for the sentence embedding from "tough to beat baseline"
        :param list[numpy.array] sentence:
        :param list[str] word_list:
        :param a: factor for word weights
        :return: preliminary embedding vector for the given sentence
        """
        vec = np.zeros((self.embedding_dim, 1))
        for w_embed, word in zip(sentence, word_list):
            temp_vec = np.reshape(w_embed, (self.embedding_dim, 1))
            if word in self.wiki_freq_dict:
                vec = vec + 100 * temp_vec * a / (a + self.wiki_freq_dict[word])
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


class FlairDataEmbedder(DataEmbedder):

    def __init__(self, embeddings, create_sent_emb=False, wiki_path=""):
        super(FlairDataEmbedder, self).__init__(create_sent_emb=create_sent_emb, wiki_path=wiki_path)
        self.embeddings = embeddings
        self.embedding_dim = self.embeddings.embedding_length

    def _embed_sentence(self, sent):
        """
        :param list[str] sent:
        :return: list[numpy.array]
        """
        s_list = []
        flair_sent = Sentence(" ".join(sent))
        self.embeddings.embed(flair_sent)
        for w in flair_sent:
            s_list.append(w.embedding.numpy())
        return s_list


class SimpleDataEmbedder(DataEmbedder):

    def __init__(self, embedding_man, create_sent_emb=False, wiki_path=""):
        """
        :param DataSet ds:
        :param EmbeddingMan embedding_man:
        """
        super(SimpleDataEmbedder, self).__init__(create_sent_emb=create_sent_emb, wiki_path=wiki_path)
        self.embedding_man = embedding_man
        self.word_set = set()
        self.embedding_dim = self.embedding_man.dim

    def _embed_sentence(self, sent):
        s_list = []
        for w in sent:
            s_list.append(self.embedding_man.get_embedding_vec(w))
        return s_list


if __name__ == "__main__":
    ds1 = DataSet("../data/standardized/conll_test.txt")
    ds1.read_data()
    ds2 = DataSet("../data/standardized/itac_test.txt")
    ds2.read_data()
    ds3 = DataSet("../data/standardized/rsics_test.txt")
    ds3.read_data()

    g_man = GloveEmbeddings("glove/glove.6B.50d.txt", 50)
    embedder = SimpleDataEmbedder(g_man, create_sent_emb=False)
    embedder.reduce_dataset(ds1, path="dim_reductions/glove50_conll.csv")
    embedder.reduce_dataset(ds2, path="dim_reductions/glove50_itac.csv")
    embedder.reduce_dataset(ds3, path="dim_reductions/glove50_rsics.csv")

    b_man = BPEmbeddings(dim=100, bp_vocab_size=50000)
    embedder = SimpleDataEmbedder(b_man, create_sent_emb=False)
    embedder.reduce_dataset(ds1, path="dim_reductions/bp-d100-vs50000_conll.csv")
    embedder.reduce_dataset(ds2, path="dim_reductions/bp-d100-vs50000_itac.csv")
    embedder.reduce_dataset(ds3, path="dim_reductions/bp-d100-vs50000_rsics.csv")

    c_man = CombinedEmbeddings([g_man, b_man])
    embedder = SimpleDataEmbedder(c_man, create_sent_emb=False)
    embedder.reduce_dataset(ds1, path="dim_reductions/bp-glove_conll.csv")
    embedder.reduce_dataset(ds2, path="dim_reductions/bp-glove_itac.csv")
    embedder.reduce_dataset(ds3, path="dim_reductions/bp-glove_rsics.csv")

    embedder = FlairDataEmbedder(WordEmbeddings("en-crawl"))
    embedder.reduce_dataset(ds1, path="dim_reductions/fasttext_en-crawl_conll.csv")
    embedder.reduce_dataset(ds2, path="dim_reductions/fasttext_en-crawl_itac.csv")
    embedder.reduce_dataset(ds3, path="dim_reductions/fasttext_en-crawl_rsics.csv")

    embedder = FlairDataEmbedder(StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]))
    embedder.reduce_dataset(ds1, path="dim_reductions/flair-forward-backward_conll.csv")
    embedder.reduce_dataset(ds2, path="dim_reductions/flair-forward-backward_itac.csv")
    embedder.reduce_dataset(ds3, path="dim_reductions/flair-forward-backward_rsics.csv")
