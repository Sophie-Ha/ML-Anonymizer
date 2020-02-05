import pandas as pd
import time
import random

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, Embeddings

from embeddings.embedding_manager import BPEmbeddings, GloveEmbeddings, CombinedEmbeddings
from pointer.ptr_manager import PointerManager
from pointer.ptr_flair_manager import FlairPointerManager
from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet

CUDA_DEVICE = 0

train_sets = ["data/standardized/conll_train.txt", "data/standardized/itac_dev.txt", "data/standardized/itac_train0.txt"]
dev_set = ["data/standardized/conll_valid.txt"]
itac_test = ["data/standardized/itac_test.txt"]
conll_test = ["data/standardized/conll_test.txt"]
rsics_test = ["data/standardized/rsics_test.txt"]

wiki_file = "embeddings/enwiki-20190320-words-frequency.txt"

glove_embeddings = [("embeddings/glove/glove.6B.50d.txt", 50),
                    ("embeddings/glove/glove.6B.100d.txt", 100),
                    ("embeddings/glove/glove.6B.200d.txt", 200),
                    ("embeddings/glove/glove.6B.300d.txt", 300)]

bpemb_vocab_sizes = [1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000]
bpemb_dims = [25, 50, 100, 200, 300]

# PN settings:
START_LR = 0.01
LR_PATIENCE = 3
LR_DECAY = 0.5
EPOCH_PATIENCE = 6
MAX_EPOCHS = 20
TEACHER_FORCING = 0.5


def pointer_test_n_layers():
    g_dim_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    epoch_list = []
    n_list = []

    d = 50
    for n_layers in range(1, 5):
        n_list.append(n_layers)
        g_dim_list.append(d)
        path = "embeddings/glove/glove.6B.{}d.txt".format(d)
        g_man = GloveEmbeddings(path=path, dim=d)

        manager = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                 lr_patience=LR_PATIENCE, n_encoder_layers=n_layers, n_decoder_layers=n_layers,
                                 cuda_device=CUDA_DEVICE)
        st = time.time()
        manager.train_model_dynamic(train_sets=train_sets, dev_sets=dev_set, max_tries=EPOCH_PATIENCE,
                                    print_interval=10000,
                                    save_path="pointer/models/19_06_06/glove_d{}_{}layers.pt".format(d, n_layers),
                                    teacher_forcing=TEACHER_FORCING)
        time_list.append(time.time() - st)
        manager2 = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                  lr_patience=LR_PATIENCE, n_encoder_layers=n_layers, n_decoder_layers=n_layers,
                                  cuda_device=CUDA_DEVICE)
        manager2.load_model("pointer/models/19_06_06/glove_d{}_{}layers.pt".format(d, n_layers))
        epoch_list.append(manager2.model.cur_epoch)
        r, p, f = manager2.test_model(itac_test)
        itac_f.append(f)
        itac_p.append(p)
        itac_r.append(r)

        r, p, f = manager2.test_model(conll_test)
        conll_f.append(f)
        conll_p.append(p)
        conll_r.append(r)
    result_df = pd.DataFrame()
    result_df["glove_dim"] = g_dim_list
    result_df["num_layers"] = n_list
    result_df["epochs_trained"] = epoch_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_06_06/ptr_glove_n_layers.csv")


def flair_pointer_test_embeddings():
    embedding_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    epoch_list = []

    def test_model(embeddings, emb_name):
        embedding_list.append(emb_name)

        man = FlairPointerManager(embeddings, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                                  n_encoder_layers=1, n_decoder_layers=1, cuda_device=CUDA_DEVICE)
        st = time.time()
        man.train_model_dynamic(train_sets=train_sets, dev_sets=dev_set, max_tries=EPOCH_PATIENCE,
                                print_interval=10000,
                                save_path="pointer/models/19_06_06/flair_{}.pt".format(emb_name),
                                teacher_forcing=TEACHER_FORCING)
        time_list.append(time.time() - st)
        man2 = FlairPointerManager(embeddings, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                                   n_encoder_layers=1, n_decoder_layers=1, cuda_device=CUDA_DEVICE)
        man2.load_model("pointer/models/19_06_06/flair_{}.pt".format(emb_name))
        epoch_list.append(man2.model.cur_epoch)
        r, p, f = man2.test_model(itac_test)
        itac_f.append(f)
        itac_p.append(p)
        itac_r.append(r)

        r, p, f = man2.test_model(conll_test)
        conll_f.append(f)
        conll_p.append(p)
        conll_r.append(r)

    test_model(WordEmbeddings("en-crawl"), "fasttext_en-crawl")
    test_model(WordEmbeddings("news"), "fasttext_news")
    test_model(FlairEmbeddings("news-forward-fast"), "flair_news-forward-fast")
    test_model(FlairEmbeddings("news-backward-fast"), "flair_news-backward-fast")
    test_model(StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]),
               "flair_news_forward-backward")
    test_model(FlairEmbeddings("mix-forward"), "flair_mix-forward")
    test_model(BertEmbeddings("bert-base-uncased"), "bert-base-uncased")

    result_df = pd.DataFrame()
    result_df["embedding_type"] = embedding_list
    result_df["epochs_trained"] = epoch_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_06_06/ptr_flair_embeddings.csv")


def ptr_test_datasets():

    train_list = []
    dev_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    rsics_p = []
    rsics_r = []
    rsics_f = []
    epoch_list = []

    def test_model(train_ds, train_string, dev_ds, dev_string, done=True):
        train_list.append(train_string)
        dev_list.append(dev_string)
        emb = StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')])
        if not done:
            man = FlairPointerManager(emb, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                                      n_encoder_layers=1, n_decoder_layers=1, cuda_device=CUDA_DEVICE)
            man.train_model_dynamic(train_sets=train_ds, dev_sets=dev_ds, max_tries=EPOCH_PATIENCE,
                                    print_interval=10000,
                                    save_path="pointer/models/19_06_06/flair_{}.pt".format(train_string),
                                    teacher_forcing=TEACHER_FORCING)

        man2 = FlairPointerManager(emb, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                                   n_encoder_layers=1, n_decoder_layers=1, cuda_device=CUDA_DEVICE)
        man2.load_model("pointer/models/19_06_06/flair_{}.pt".format(train_string))
        time_list.append(man2.model.train_time)

        epoch_list.append(man2.model.cur_epoch)
        r, p, f = man2.test_model(itac_test)
        itac_f.append(f)
        itac_p.append(p)
        itac_r.append(r)

        r, p, f = man2.test_model(conll_test)
        conll_f.append(f)
        conll_p.append(p)
        conll_r.append(r)

        r, p, f = man2.test_model(rsics_test)
        rsics_p.append(p)
        rsics_r.append(r)
        rsics_f.append(f)

    test_model(train_ds=train_sets+["data/standardized/rsics_train.txt", "data/standardized/rsics_dev.txt"],
               train_string="conll+itac+rsics",
               dev_ds=dev_set, dev_string="conll", done=True)
    print("\nTrained 1 out of 7 models\n")
    test_model(train_ds=["data/standardized/rsics_train.txt", "data/standardized/itac_dev.txt", "data/standardized/itac_train0.txt"],
               train_string="itac+rsics",
               dev_ds=["data/standardized/rsics_dev.txt"], dev_string="rsics")
    print("\nTrained 2 out of 7 models\n")
    test_model(train_ds=train_sets,
               train_string="itac+conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 3 out of 7 models\n")
    test_model(train_ds=["data/standardized/rsics_train.txt", "data/standardized/conll_train.txt",
                         "data/standardized/rsics_dev.txt"],
               train_string="rsics+conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 4 out of 7 models\n")
    test_model(train_ds=["data/standardized/rsics_train.txt"],
               train_string="rsics",
               dev_ds=["data/standardized/rsics_dev.txt"], dev_string="rsics")
    print("\nTrained 5 out of 7 models\n")
    test_model(train_ds=["data/standardized/itac_train0.txt"],
               train_string="itac",
               dev_ds=["data/standardized/itac_dev.txt"], dev_string="itac")
    print("\nTrained 6 out of 7 models\n")
    test_model(train_ds=["data/standardized/conll_train.txt"],
               train_string="conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 7 out of 7 models\n")

    result_df = pd.DataFrame()
    result_df["train_sets"] = train_list
    result_df["dev_sets"] = dev_list
    result_df["epochs_trained"] = epoch_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df["precision_rsics"] = rsics_p
    result_df["recall_rsics"] = rsics_r
    result_df["f1_rsics"] = rsics_f
    result_df.to_csv("results/19_06_06/ptr_flair_datasets.csv")


def bc_test_datasets():
    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier", "LinearSVC", "RandomForest",
                 "GradientBoosting"]

    model_list = []
    train_list = []
    dev_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    rsics_p = []
    rsics_r = []
    rsics_f = []

    embeddings = StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')])

    def test_model(train_ds, train_string, dev_ds, dev_string):
        classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                       SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                       RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

        for clf, clf_name in zip(classifiers, clf_names):
            model_list.append(clf_name)
            train_list.append(train_string)
            dev_list.append(dev_string)

            model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file)

            st = time.time()
            model.train_model(data_files=train_ds+dev_ds)
            time_list.append(time.time() - st)

            r, p, f = model.test_model(itac_test)
            itac_f.append(f)
            itac_p.append(p)
            itac_r.append(r)

            r, p, f = model.test_model(conll_test)
            conll_f.append(f)
            conll_p.append(p)
            conll_r.append(r)

            r, p, f = model.test_model(rsics_test)
            rsics_f.append(f)
            rsics_p.append(p)
            rsics_r.append(r)

            result_df = pd.DataFrame()
            result_df["clf_type"] = model_list
            result_df["train_sets"] = train_list
            result_df["dev_sets"] = dev_list
            result_df["train_time"] = time_list
            result_df["precision_itac"] = itac_p
            result_df["recall_itac"] = itac_r
            result_df["f1_itac"] = itac_f
            result_df["precision_conll"] = conll_p
            result_df["recall_conll"] = conll_r
            result_df["f1_conll"] = conll_f
            result_df["precision_rsics"] = rsics_p
            result_df["recall_rsics"] = rsics_r
            result_df["f1_rsics"] = rsics_f
            result_df.to_csv("results/19_06_06/bc_flair_datasets.csv")

    test_model(train_ds=train_sets + ["data/standardized/rsics_train.txt", "data/standardized/rsics_dev.txt"],
               train_string="conll+itac+rsics",
               dev_ds=dev_set, dev_string="conll")
    print("\nTrained 1 out of 7 model sets\n")
    test_model(train_ds=["data/standardized/rsics_train.txt", "data/standardized/itac_dev.txt",
                         "data/standardized/itac_train0.txt"],
               train_string="itac+rsics",
               dev_ds=["data/standardized/rsics_dev.txt"], dev_string="rsics")
    print("\nTrained 2 out of 7 model sets\n")
    test_model(train_ds=train_sets,
               train_string="itac+conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 3 out of 7 model sets\n")
    test_model(train_ds=["data/standardized/rsics_train.txt", "data/standardized/conll_train.txt",
                         "data/standardized/rsics_dev.txt"],
               train_string="rsics+conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 4 out of 7 model sets\n")
    test_model(train_ds=["data/standardized/rsics_train.txt"],
               train_string="rsics",
               dev_ds=["data/standardized/rsics_dev.txt"], dev_string="rsics")
    print("\nTrained 5 out of 7 model sets\n")
    test_model(train_ds=["data/standardized/itac_train0.txt"],
               train_string="itac",
               dev_ds=["data/standardized/itac_dev.txt"], dev_string="itac")
    print("\nTrained 6 out of 7 model sets\n")
    test_model(train_ds=["data/standardized/conll_train.txt"],
               train_string="conll",
               dev_ds=["data/standardized/conll_valid.txt"], dev_string="conll")
    print("\nTrained 7 out of 7 model sets\n")


def bc_test_flair_embeddings():
    # clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier",
    #             "BayesianGaussianMixture", "GaussianNB", "LinearSVC", "RandomForest", "GradientBoosting"]

    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier", "LinearSVC", "RandomForest",
                 "GradientBoosting"]

    embedding_list = []
    model_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []

    def test_models(embeddings, emb_type):
        # classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
        #               SGDClassifier(class_weight="balanced"), BayesianGaussianMixture(), GaussianNB(),
        #               LinearSVC(class_weight="balanced"), RandomForestClassifier(class_weight="balanced"),
        #               GradientBoostingClassifier()]

        classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                       SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                       RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

        for clf, clf_name in zip(classifiers, clf_names):
            embedding_list.append(emb_type)
            model_list.append(clf_name)

            model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file)
            st = time.time()
            model.train_model(data_files=train_sets)
            time_list.append(time.time()-st)

            r, p, f = model.test_model(itac_test)
            itac_f.append(f)
            itac_p.append(p)
            itac_r.append(r)

            r, p, f = model.test_model(conll_test)
            conll_f.append(f)
            conll_p.append(p)
            conll_r.append(r)

        result_df = pd.DataFrame()
        result_df["embedding_type"] = embedding_list
        result_df["clf_type"] = model_list
        result_df["train_time"] = time_list
        result_df["precision_itac"] = itac_p
        result_df["recall_itac"] = itac_r
        result_df["f1_itac"] = itac_f
        result_df["precision_conll"] = conll_p
        result_df["recall_conll"] = conll_r
        result_df["f1_conll"] = conll_f
        result_df.to_csv("results/19_06_06/bc_flair_embeddings2.csv")

    # test_models(WordEmbeddings("en-crawl"), "fasttext_en-crawl")
    # print("performed 1 out of 7 tests")
    # test_models(WordEmbeddings("news"), "fasttext_news")
    # print("performed 2 out of 7 tests")
    # test_models(FlairEmbeddings("news-forward-fast"), "flair_news-forward-fast")
    # print("performed 3 out of 7 tests")
    # test_models(FlairEmbeddings("news-backward-fast"), "flair_news-backward-fast")
    # print("performed 4 out of 7 tests")
    # test_models(StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')]),
    #             "flair_news_forward-backward")
    # print("performed 5 out of 7 tests")
    test_models(FlairEmbeddings("mix-forward"), "flair_mix-forward")
    print("performed 6 out of 7 tests")
    test_models(BertEmbeddings("bert-base-uncased"), "bert-base-uncased")
    print("performed 7 out of 7 tests")


def download_flair_models():
    w = WordEmbeddings("en-crawl")
    w = WordEmbeddings("news")
    w = FlairEmbeddings("news-forward-fast")
    w = FlairEmbeddings("news-backward-fast")
    w = FlairEmbeddings("mix-forward")
    w = BertEmbeddings("bert-base-uncased")


def get_dataset_stats():
    conll = DataSet("")
    conll.read_multiple(["data/standardized/conll_train.txt", "data/standardized/conll_test.txt", "data/standardized/conll_valid.txt"])
    print("CoNLL n words: {}; n annotations: {}; n sentences: {}".format(conll.word_count, conll.annotation_count,
                                                                         conll.sentence_count))

    itac = DataSet("")
    itac.read_multiple(["data/standardized/itac_dev.txt", "data/standardized/itac_train0.txt", "data/standardized/itac_test.txt"])
    print("ITAC n words: {}; n annotations: {}; n sentences: {}".format(itac.word_count, itac.annotation_count,
                                                                        itac.sentence_count))

    rsics = DataSet("data/standardized/rsics_complete.txt")
    rsics.read_data()
    print("RSICS n words: {}; n annotations: {}; n sentences: {}".format(rsics.word_count, rsics.annotation_count,
                                                                         rsics.sentence_count))


if __name__ == "__main__":
    # pointer_test_n_layers()
    # get_dataset_stats()
    # print("\n++++Performing two sets of experiments, training 7 models each+++++\n")
    # download_flair_models()
    # flair_pointer_test_embeddings()
    bc_test_flair_embeddings()
    # ptr_test_datasets()
    # bc_test_datasets()
