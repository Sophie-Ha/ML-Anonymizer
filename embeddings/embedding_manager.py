import numpy as np
import torch
from bpemb import BPEmb
from flair.embeddings import WordEmbeddings
from flair.data import Sentence

from dataset import DataSet


class EmbeddingMan:

    def __init__(self, case_sensitive=False, dim=None):
        self.word2id = dict()
        self.id2word = dict()
        self.vocab_size = 0
        self.embedding_tensor = None
        self.dim = dim
        self.case_sensitive = case_sensitive

    def get_embedding_vec(self, word):
        """
        returns embedding vector of a single word
        :param word:
        :return: numpy array containing the embedding
        """
        if self.case_sensitive:
            if word in self.word2id:
                id = self.word2id[word]
            else:
                id = self.word2id["<UNK>"]
            return self.embedding_tensor.numpy()[id]
        else:
            if word.lower() in self.word2id:
                id = self.word2id[word.lower()]
            else:
                id = self.word2id["<UNK>"]
            return self.embedding_tensor.numpy()[id]

    def get_word_by_id(self, key):
        if key not in self.id2word:
            raise KeyError("Word id {} does not exist in id2word dictionary".format(key))
        return self.id2word[key]

    def get_id_by_word(self, word):
        if self.case_sensitive:
            if word in self.word2id:
                return self.word2id[word]
            else:
                return self.word2id["<UNK>"]
        else:
            if word.lower() in self.word2id:
                return self.word2id[word.lower()]
            else:
                return self.word2id["<UNK>"]

    def build_vocabulary(self, datasets):
        """
        Creates embeddings for all words in the dataset
        :param list[DataSet] datasets:
        :return:
        """

        embedding_list = []

        for ds in datasets:
            for doc in ds.documents:
                for sent in doc.sentences:
                    for word in sent:
                        if not self.case_sensitive:
                            word = word.lower()
                        if word not in self.word2id:
                            vec = self.get_embedding_vec(word)
                            embedding_list.append(vec)
                            self.word2id[word] = self.vocab_size
                            self.id2word[self.vocab_size] = word
                            self.vocab_size += 1

        self.word2id["<EOS>"] = self.vocab_size
        self.id2word[self.vocab_size] = "<EOS>"
        embedding_list.append(np.ones((self.dim, )))
        self.vocab_size += 1

        self.word2id["<UNK>"] = self.vocab_size
        self.id2word[self.vocab_size] = "<UNK>"
        embedding_list.append(np.zeros((self.dim,)))
        self.vocab_size += 1

        self.word2id["<SOS>"] = self.vocab_size
        self.id2word[self.vocab_size] = "<SOS>"
        embedding_list.append(np.full((self.dim, ), -1.))
        self.vocab_size += 1

        embedding_array = np.stack(embedding_list, axis=0)
        self.embedding_tensor = torch.from_numpy(embedding_array)


class FlairWordEmbeddings(EmbeddingMan):

    def __init__(self, flairembeddings, case_sensitive=False):
        """

        :param WordEmbeddings flairembeddings:
        :param case_sensitive:
        """
        self.embeddings = flairembeddings
        dim = flairembeddings.embedding_length
        super(FlairWordEmbeddings, self).__init__(case_sensitive=case_sensitive, dim=dim)

    def get_embedding_vec(self, word):
        sent = Sentence(word)
        self.embeddings.embed(sent)
        return np.reshape(sent[0].embedding.numpy(), (self.dim, ))


class CombinedEmbeddings(EmbeddingMan):

    def __init__(self, emb_mans, case_sensitive=False):
        self.emb_mans = emb_mans
        dim = sum([e.dim for e in emb_mans])
        super(CombinedEmbeddings, self).__init__(case_sensitive=case_sensitive, dim=dim)

    def get_embedding_vec(self, word):
        vecs = []
        for man in self.emb_mans:
            vecs.append(man.get_embedding_vec(word))
        return np.reshape(np.concatenate(vecs), (self.dim, ))


class BPEmbeddings(EmbeddingMan):

    def __init__(self, case_sensitive=False, dim=None, bp_vocab_size=0):
        super(BPEmbeddings, self).__init__(case_sensitive=case_sensitive, dim=dim)
        self.bp_vocab_size = bp_vocab_size
        self.model = BPEmb(lang="en", dim=self.dim, vs=self.bp_vocab_size)

    def get_embedding_vec(self, word):
        if self.model is None:
            self.model = BPEmb(lang="en", dim=self.dim, vs=self.bp_vocab_size)
        if not self.case_sensitive:
            word = word.lower()
        vecs = self.model.embed(word)
        return np.reshape(np.sum(vecs, axis=0), (self.dim, ))


class GloveEmbeddings(EmbeddingMan):

    def __init__(self, path, dim, case_sensitive=False):
        super(GloveEmbeddings, self).__init__(case_sensitive=case_sensitive, dim=dim)
        self.load(path, dim)

    def load(self, path, dim):
        """
        loads in word vectors
        :param path: path to the saved vectors
        :param dim: dimension of the embeddings
        :return:
        """
        # Requires: http://nlp.stanford.edu/data/glove.6B.zip
        # First load in glove
        with open(path, encoding="utf8") as f:
            content = f.readlines()

        embeddings = np.zeros((len(content) + 3, dim))
        self.dim = dim

        for word_line, c in zip(content, range(len(content))):
            data = word_line.split()
            word = data[0]
            vec = np.reshape(np.asarray([float(x) for x in data[1:]]), newshape=(len(data) - 1,))
            if self.case_sensitive:
                self.word2id[word] = c
                self.id2word[c] = word
            else:
                self.word2id[word.lower()] = c
                self.id2word[c] = word.lower()
            embeddings[c] = vec

        c = len(self.id2word)
        self.word2id["<EOS>"] = c
        self.id2word[c] = "<EOS>"
        embeddings[c] = np.ones((1, dim))

        self.word2id["<UNK>"] = c+1
        self.id2word[c+1] = "<UNK>"

        self.word2id["<SOS>"] = c+2
        self.id2word[c+2] = "<SOS>"
        embeddings[c+2] = np.full((1, dim), -1.)

        self.embedding_tensor = torch.from_numpy(embeddings)


if __name__ == "__main__":
    bp_man = BPEmbeddings(bp_vocab_size=3000, dim=50)
    glove_man = GloveEmbeddings("../embeddings/glove.6B.50d.txt", dim=50)
    c_man = CombinedEmbeddings([glove_man, bp_man])
    b_vec = bp_man.get_embedding_vec("Sophie")
    g_vec = glove_man.get_embedding_vec("Sophie")
    c_vec = c_man.get_embedding_vec("Sophie")

    ds = DataSet("../data/standardized/conll_valid.txt")
    ds.read_multiple(["../data/standardized/temp/rsics_1_2.txt", "../data/standardized/temp/rsics_1_2.txt_manual_done.txt",
                      "../data/standardized/temp/rsics_1_3.txt", "../data/standardized/temp/rsics_1_3.txt_manual_done.txt",
                      "../data/standardized/temp/rsics_1_4.txt", "../data/standardized/temp/rsics_1_4.txt_manual_done.txt"])
    bp_man.build_vocabulary([ds])
    print(bp_man.vocab_size)
    print("Done")

