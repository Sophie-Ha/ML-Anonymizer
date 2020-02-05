

#  Download n-gram frequency lists from https://www.ngrams.info/download_coca.asp (non case-sensitive)


from dataset import DataSet, Document
from evaluation.metrics import MetricHelper


class NGramStorage:

    def __init__(self, n, case_sensitive=False, path=""):
        self.n = n
        self.case_sensitive = case_sensitive
        self.word_set = set()
        if path:
            self.load_data(path)

    def load_data(self, path):
        with open(path, encoding="ISO-8859-1") as f:
            for line in f.readlines():
                w_list = line.split()
                self.word_set.add(" ".join(w_list[1:]))

    def check_word(self, w):
        if self.case_sensitive:
            word = w
        else:
            word = w.lower()
        return word in self.word_set


class CorpusBaselineManager:

    def __init__(self, ngram):
        self.ngram = ngram

    def annotate_doc(self, doc):
        """

        :param Document doc:
        :return:
        """
        predicted_annotations = []
        n = self.ngram.n

        for sent in doc.sentences:
            sent_anno = [0] * len(sent)
            for i in range(len(sent) - n + 1):
                w = " ".join(sent[i:i+n])
                if not self.ngram.check_word(w):
                    for j in range(i, i+n):
                        sent_anno[j] = 1
            predicted_annotations.append(sent_anno)
        return predicted_annotations

    def test_ngram(self, datasets):
        ds = DataSet(datasets[0])
        ds.read_multiple(datasets)
        predicted_list = []

        for doc in ds.documents:
            predicted = self.annotate_doc(doc)
            predicted_list += predicted
        mh = MetricHelper(predicted=predicted_list, target=ds.get_merged_annotations())
        print("Recall: {}, Precision: {}, F1: {}".format(mh.recall(), mh.precision(), mh.f1()))


if __name__ == "__main__":
    n_storage = NGramStorage(n=2)
    n_storage.load_data("n-gram_frequencies/w2_.txt")

    print("\n\n n=2 \n")
    corpus_man = CorpusBaselineManager(n_storage)
    print("\nconll_train:")
    corpus_man.test_ngram(["../data/standardized/conll_train.txt"])
    print("\nconll_valid:")
    corpus_man.test_ngram(["../data/standardized/conll_valid.txt"])
    print("\nconll_test:")
    corpus_man.test_ngram(["../data/standardized/conll_test.txt"])
    print("\nitac_dev:")
    corpus_man.test_ngram(["../data/standardized/itac_dev.txt"])
    print("\nitac_test:")
    corpus_man.test_ngram(["../data/standardized/itac_test.txt"])

    n_storage = NGramStorage(n=3)
    n_storage.load_data("n-gram_frequencies/w3_.txt")

    print("\n\n n=3 \n")
    corpus_man = CorpusBaselineManager(n_storage)
    print("\nconll_train:")
    corpus_man.test_ngram(["../data/standardized/conll_train.txt"])
    print("\nconll_valid:")
    corpus_man.test_ngram(["../data/standardized/conll_valid.txt"])
    print("\nconll_test:")
    corpus_man.test_ngram(["../data/standardized/conll_test.txt"])
    print("\nitac_dev:")
    corpus_man.test_ngram(["../data/standardized/itac_dev.txt"])
    print("\nitac_test:")
    corpus_man.test_ngram(["../data/standardized/itac_test.txt"])

    n_storage = NGramStorage(n=4)
    n_storage.load_data("n-gram_frequencies/w4_.txt")

    print("\n\n n=4 \n")
    corpus_man = CorpusBaselineManager(n_storage)
    print("\nconll_train:")
    corpus_man.test_ngram(["../data/standardized/conll_train.txt"])
    print("\nconll_valid:")
    corpus_man.test_ngram(["../data/standardized/conll_valid.txt"])
    print("\nconll_test:")
    corpus_man.test_ngram(["../data/standardized/conll_test.txt"])
    print("\nitac_dev:")
    corpus_man.test_ngram(["../data/standardized/itac_dev.txt"])
    print("\nitac_test:")
    corpus_man.test_ngram(["../data/standardized/itac_test.txt"])
