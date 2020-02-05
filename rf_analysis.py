import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier

from embeddings.embedding_manager import BPEmbeddings, GloveEmbeddings, CombinedEmbeddings
from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet

train_sets = ["data/standardized/conll_train.txt", "data/standardized/itac_train0.txt",
              "data/standardized/itac_dev.txt"]
dev_set = ["data/standardized/conll_valid.txt"]
itac_test = ["data/standardized/itac_test.txt"]
conll_test = ["data/standardized/conll_test.txt"]

wiki_file = "embeddings/enwiki-20190320-words-frequency.txt"

glove_embeddings = [("embeddings/glove/glove.6B.50d.txt", 50),
                    ("embeddings/glove/glove.6B.100d.txt", 100),
                    ("embeddings/glove/glove.6B.200d.txt", 200),
                    ("embeddings/glove/glove.6B.300d.txt", 300)]


def test_depth_n_trees():
    d_list = []
    n_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []

    for d in list(range(5, 20)):  # [5, 10, 20, 50, None]:
        for n in [10, 20, 50, 100]:
            d_list.append(d)
            n_list.append(n)
            print("\nTraining {} trees with max_depth {}".format(n, d))
            g_man = GloveEmbeddings(path="embeddings/glove/glove.6B.50d.txt", dim=50)
            clf = BasicClassifier(model=RandomForestClassifier(n_estimators=n, max_depth=d),
                                  emb_man=g_man, wiki_file=wiki_file)
            t = time.time()
            clf.train_model(data_files=train_sets)
            time_list.append(time.time()-t)

            r, p, f = clf.test_model(itac_test)
            itac_f.append(f)
            itac_p.append(p)
            itac_r.append(r)

            r, p, f = clf.test_model(conll_test)
            conll_f.append(f)
            conll_p.append(p)
            conll_r.append(r)

    result_df = pd.DataFrame()
    result_df["n_estimators"] = n_list
    result_df["max_depth"] = d_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/random_forest/n_trees_depth_search2.csv")


if __name__ == "__main__":
    test_depth_n_trees()
