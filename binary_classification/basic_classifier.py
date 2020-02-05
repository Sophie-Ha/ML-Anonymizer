import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from flair.embeddings import Embeddings
from flair.data import Sentence

from dataset import DataSet, Document
from embeddings.embedding_processing import EmbeddingProcessor
from embeddings.embedding_manager import GloveEmbeddings
from evaluation.metrics import MetricHelper
from evaluation.misclassifications import analyze_misclassifications


class BasicClassifier:

    def __init__(self, model, emb_man, emb_type="standard", lowercase=False,
                 wiki_file="../embeddings/enwiki-20150602-words-frequency.txt", embed_sentences=False):

        self.model = model
        if emb_type == "standard":
            self.data_helper = DataHelper(emb_man=emb_man, wiki_file=wiki_file, lowercase=lowercase)
        elif emb_type == "flair":
            self.data_helper = FlairDataHelper(embeddings=emb_man, lowercase=lowercase, embed_sentences=embed_sentences,
                                               wiki_path=wiki_file)
        else:
            raise ValueError("embedding type {} does not exist".format(emb_type))

    def train_model(self, data_files):
        train_ds = DataSet(path=data_files[0])
        train_ds.read_multiple(data_files)
        t_data, target = self.data_helper.build_train_data(train_ds)
        self.model.fit(t_data, target)
        print("Done Training")

    def test_model(self, data_files):
        test_ds = DataSet(path=data_files[0])
        test_ds.read_multiple(data_files)
        t_data, target = self.data_helper.build_train_data(test_ds)

        predicted = self.model.predict(t_data)
        mh = MetricHelper(predicted=[predicted], target=[target])
        print("Recall: {}, Precision: {}, F1: {}".format(mh.recall(), mh.precision(), mh.f1()))
        return mh.recall(), mh.precision(), mh.f1()

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

    def _annotate_document(self, doc):
        ds = DataSet("")
        ds.add_document(doc)
        embedded = self.data_helper.embed_dataset(ds)[0]

        predicted = []
        for sent_emb in embedded:
            data = np.array(sent_emb)
            p = self.model.predict(data)
            predicted.append(p)
        return predicted

    def test_misclassifications(self, datasets, csv_path):
        ds = DataSet(datasets[0], reduce_to_one=True)
        ds.read_multiple(datasets)

        for doc in ds.documents:
            doc.predicted = self._annotate_document(doc)

        df = analyze_misclassifications(ds)
        df.to_csv(csv_path)

    def annotate_string(self, input_text):
        doc = Document(raw_text=input_text)
        doc.create_from_text()
        ds = DataSet("")
        ds.add_document(doc)
        embedded = self.data_helper.embed_dataset(ds)[0]

        predicted = []
        for sent_emb in embedded:
            data = np.array(sent_emb)
            p = self.model.predict(data)
            predicted.append(p)
        doc.annotated = predicted
        return doc.create_text()


class DataHelper:

    def __init__(self, emb_man, wiki_file="../embeddings/enwiki-20150602-words-frequency.txt", lowercase=False):

        self.emb_man = emb_man
        self.emb_processor = EmbeddingProcessor(self.emb_man, wiki_file)
        self.lowercase = lowercase

    def embed_dataset(self, ds):
        """
        Prepares embeddings consisting of concatenated word and sentence vector
        :param DataSet ds: dataset for which the embeddings are created
        :return: list[list[numpy array]] list of embedding vectors per sentence
        """

        ds_embs = []

        for doc in ds.documents:
            doc_embs = []
            sent_embs = self.emb_processor.get_sent_emb_simple(doc)

            for sent, emb in zip(doc.sentences, sent_embs):
                s_list = []

                for w in sent:
                    w_emb = np.concatenate((np.reshape(self.emb_man.get_embedding_vec(w), (self.emb_man.dim,)),
                                            np.reshape(emb, (self.emb_man.dim,))))
                    s_list.append(w_emb)
                doc_embs.append(s_list)
            ds_embs.append(doc_embs)

        return ds_embs

    def build_train_data(self, ds):
        """

        :param DataSet ds:
        :return:
        """
        embeddings = self.embed_dataset(ds)
        emb_vecs = []
        targets = []

        for doc, doc_emb in zip(ds.documents, embeddings):
            for sent, sent_embs in zip(doc.annotated, doc_emb):
                emb_vecs += sent_embs
                targets += sent

        data = np.array(emb_vecs)
        print("Training data has shape", data.shape)

        assert data.shape[0] == len(targets), "Number of targets does not match the shape of the training data"

        return data, targets


class FlairDataHelper:

    def __init__(self, embeddings, embed_sentences=False, wiki_path=None, lowercase=False):
        self.embeddings = embeddings
        self.embed_sentences = embed_sentences
        self.wiki_path = wiki_path
        self.wiki_freq_dict = None
        self.total_wiki_dict_count = 0
        self.lowercase = lowercase
        if embed_sentences:
            assert wiki_path is not None, "Need path to the wiki frequency dict to create sentence embeddings"

    def embed_dataset(self, ds):
        """
        Prepares embeddings consisting of concatenated word and sentence vector
        :param DataSet ds: dataset for which the embeddings are created
        :return: list[list[numpy array]] list of embedding vectors per sentence
        """

        ds_embs = []
        for doc in ds.documents:
            doc_embs = []
            if self.embed_sentences:
                sent_embs = self.get_sent_emb_simple(doc)
            i = 0
            for sent in doc.sentences:
                s_list = []

                if self.lowercase:
                    flair_sent = Sentence(" ".join([s.lower() for s in sent]))
                else:
                    flair_sent = Sentence(" ".join(sent))
                self.embeddings.embed(flair_sent)

                for w in flair_sent:
                    if self.embed_sentences:
                        w_emb = np.concatenate((np.reshape(w.embedding.numpy(), (self.embeddings.embedding_length, )),
                                                np.reshape(sent_embs[i], (self.embeddings.embedding_length, ))))
                        s_list.append(w_emb)
                    else:
                        s_list.append(w.embedding.numpy())
                doc_embs.append(s_list)
                i += 1
            ds_embs.append(doc_embs)

        return ds_embs

    def get_sent_emb_simple(self, doc):
        """
        Creates sentence embeddings according to arora2016
        :param doc: Document object
        :return: list of sentence embedding vectors
        """

        if self.wiki_freq_dict is None:
            self._load_in_wiki_dic(self.wiki_path)

        sent_vecs = np.zeros((self.embeddings.embedding_length, len(doc.sentences)))

        for i in range(len(doc.sentences)):
            res = self._get_simple_sentence_vec(doc.sentences[i])
            sent_vecs[:, i] = res.reshape(self.embeddings.embedding_length)

        u, _, _ = np.linalg.svd(sent_vecs)
        svd_mat = np.matmul(u[:, 0].reshape((self.embeddings.embedding_length, 1)),
                            u[:, 0].reshape((self.embeddings.embedding_length, 1)).transpose())

        sentence_embeddings = []
        for i in range(len(doc.sentences)):
            vec = sent_vecs[:, i].reshape((self.embeddings.embedding_length, 1)) - \
                  np.matmul(svd_mat, sent_vecs[:, i].reshape((self.embeddings.embedding_length, 1)))
            sentence_embeddings.append(vec)

        return sentence_embeddings

    def build_train_data(self, ds):
        """

        :param DataSet ds:
        :return:
        """
        embeddings = self.embed_dataset(ds)
        emb_vecs = []
        targets = []

        for doc, doc_emb in zip(ds.documents, embeddings):
            for sent, sent_embs in zip(doc.annotated, doc_emb):
                emb_vecs += sent_embs
                targets += sent

        data = np.array(emb_vecs)
        print("Training data has shape", data.shape)

        assert data.shape[0] == len(targets), "Number of targets does not match the shape of the training data"

        return data, targets

    def _get_simple_sentence_vec(self, sentence, a=0.000000000001):
        """
        creates weighted sentence vector needed for the sentence embedding from "tough to beat baseline"
        :param sentence: word list from one sentence
        :param a: factor for word weights
        :return: preliminary embedding vector for the given sentence
        """
        vec = np.zeros((self.embeddings.embedding_length, 1))
        flair_sent = Sentence(" ".join(sentence))
        self.embeddings.embed(flair_sent)
        for w in flair_sent:
            temp_vec = np.reshape(w.embedding.numpy(), (self.embeddings.embedding_length, 1))
            if w.text in self.wiki_freq_dict:
                vec = vec + 100 * temp_vec * a / (a + self.wiki_freq_dict[w.text])
            else:
                vec = vec + 100 * temp_vec * a / (a + 1)
        return vec / len(sentence)

    def _load_in_wiki_dic(self, path):

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


def test_all_classifiers():
    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), BayesianGaussianMixture(), GaussianNB(),
                   LinearSVC(class_weight="balanced"), RandomForestClassifier(class_weight="balanced"),
                   GradientBoostingClassifier()]
    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier",
                 "BayesianGaussianMixture", "GaussianNB", "LinearSVC", "RandomForest", "GradientBoosting"]
    train_sets = ["../data/standardized/conll_train.txt", "../data/standardized/conll_test.txt",
                  "../data/standardized/itac_dev.txt"]
    dev_set = ["../data/standardized/conll_valid.txt"]

    for model, model_name in zip(classifiers, clf_names):
        print("\n", model_name)
        clf = BasicClassifier(model=model)
        clf.train_model(data_files=train_sets)
        clf.test_model(data_files=dev_set)
        clf.test_model(data_files=["../data/standardized/itac_test.txt"])
