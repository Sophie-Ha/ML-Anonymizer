import pandas as pd
import time

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from embeddings.embedding_manager import BPEmbeddings, GloveEmbeddings, CombinedEmbeddings
from pointer.ptr_manager import PointerManager
from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet

CUDA_DEVICE = 1

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

bpemb_vocab_sizes = [10000, 25000, 50000, 100000, 200000]  # [1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000]
bpemb_dims = [25, 50, 100, 200, 300]

# PN settings:
START_LR = 0.01
LR_PATIENCE = 3
LR_DECAY = 0.5
EPOCH_PATIENCE = 6
MAX_EPOCHS = 20
TEACHER_FORCING = 0.5


def load_bpemb_models():
    for vs in bpemb_vocab_sizes:
        for d in bpemb_dims:
            bp_man = BPEmbeddings(bp_vocab_size=vs, dim=d)


def test_example(model):
    with open("data/examples.txt") as f:
        text = f.read()
    return model.annotate_string(text)


def write_results(path, results, names=None):
    with open(path, mode="w") as f:
        for i, text in enumerate(results):
            if names is not None:
                f.write("\n\n" + names[i] + "\n")
            f.write(text)


def create_pointer_examples():
    results = []
    result_names = []
    # pure BPEmb
    vs = 100000
    d = 200
    bp_man = BPEmbeddings(bp_vocab_size=vs, dim=d, case_sensitive=False)
    ds = DataSet("blah")
    ds.read_multiple(train_sets + dev_set + itac_test + conll_test)
    bp_man.build_vocabulary([ds])
    manager = PointerManager(bp_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                             lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
    manager.load_model("pointer/models/19_05_11b/bpemb_{}_{}.pt".format(vs, d))
    results.append(test_example(manager))
    result_names.append("bpemb_{}_{}".format(vs, d))

    # pure glove
    for d in [50, 300]:
        path = "embeddings/glove/glove.6B.{}d.txt".format(d)
        g_man = GloveEmbeddings(path=path, dim=d)
        manager = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                 lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
        manager.load_model("pointer/models/19_05_11b/glove_{}.pt".format(d))
        results.append(test_example(manager))
        result_names.append("glove_{}".format(d))

    # glove + bpemb
    for g_d, b_d in [(200, 50), (300, 25)]:
        path = "embeddings/glove/glove.6B.{}d.txt".format(g_d)
        g_man = GloveEmbeddings(path=path, dim=g_d)
        b_man = BPEmbeddings(dim=b_d, bp_vocab_size=100000)
        c_man = CombinedEmbeddings([g_man, b_man])
        ds = DataSet("blah")
        ds.read_multiple(train_sets + dev_set + itac_test + conll_test)
        c_man.build_vocabulary([ds])

        manager = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                 lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
        manager.load_model("pointer/models/19_05_11b/glove_d{}_bp_d{}.pt".format(g_d, b_d))
        results.append(test_example(manager))
        result_names.append("glove_d{}_bp_d{}_vs100000".format(g_d, b_d))

    write_results("results/19_05_11b/pointer_examples.txt", results=results, names=result_names)


def pointer_with_pure_bpemb():

    # done_list = [(1000, 100), (1000, 200), (1000, 25), (1000, 50)]
    done_list = [(10000, 100), (10000, 200), (10000, 25), (10000, 300), (10000, 50), (25000, 100), (25000, 200),
                 (25000, 25), (25000, 300), (25000, 50), (50000, 100), (50000, 200), (50000, 25), (50000, 300), (50000, 50),
                 (100000, 25)]

    vs_list = []
    d_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    epoch_list = []

    for vs in bpemb_vocab_sizes:
        for d in bpemb_dims:
            vs_list.append(vs)
            d_list.append(d)
            bp_man = BPEmbeddings(bp_vocab_size=vs, dim=d, case_sensitive=False)
            ds = DataSet("blah")
            ds.read_multiple(train_sets + dev_set + itac_test + conll_test)
            bp_man.build_vocabulary([ds])

            if (vs, d) in done_list:
                manager2 = PointerManager(bp_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                          lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
                manager2.load_model("pointer/models/19_05_11b/bpemb_{}_{}.pt".format(vs, d))
            else:
                manager = PointerManager(bp_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                         lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
                # st = time.time()
                manager.train_model_dynamic(train_sets=train_sets, dev_sets=dev_set, max_tries=EPOCH_PATIENCE,
                                            print_interval=10000,
                                            save_path="pointer/models/19_05_11b/bpemb_{}_{}.pt".format(vs, d),
                                            teacher_forcing=TEACHER_FORCING)
                # time_list.append(time.time() - st)
                manager2 = PointerManager(bp_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                          lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
                manager2.load_model("pointer/models/19_05_11b/bpemb_{}_{}.pt".format(vs, d))

            time_list.append(manager2.model.train_time)
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
    result_df["vocabulary_size"] = vs_list
    result_df["bpemb dim"] = d_list
    result_df["epochs_trained"] = epoch_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_05_11b/ptr_pure_bpemb_grid_search.csv")


def pointer_with_pure_glove():
    g_dim_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    epoch_list = []

    for path, d in glove_embeddings:
        g_dim_list.append(d)
        g_man = GloveEmbeddings(path=path, dim=d)

        manager = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                 lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
        st = time.time()
        manager.train_model_dynamic(train_sets=train_sets, dev_sets=dev_set, max_tries=EPOCH_PATIENCE,
                                    print_interval=10000,
                                    save_path="pointer/models/19_05_11b/glove_{}.pt".format(d),
                                    teacher_forcing=TEACHER_FORCING)
        time_list.append(time.time() - st)
        manager2 = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                  lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
        manager2.load_model("pointer/models/19_05_11b/glove_{}.pt".format(d))
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
    result_df["glove dim"] = g_dim_list
    result_df["epochs_trained"] = epoch_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_05_11b/ptr_pure_glove_dim_search.csv")


def pointer_bpemb_glove():  # Fix Vocabulary Size to 100000
    epoch_list = []
    g_dim_list = []
    b_dim_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []

    done_list = [(100, 100), (100, 200), (100, 25), (100, 300), (100, 50), (50, 100), (50, 200), (50, 25), (50, 300),
                 (50, 50), (200, 25), (200, 50)]

    for path, d in glove_embeddings:
        for b_dim in bpemb_dims:

            g_dim_list.append(d)
            b_dim_list.append(b_dim)

            g_man = GloveEmbeddings(path=path, dim=d)
            b_man = BPEmbeddings(dim=b_dim, bp_vocab_size=100000)
            c_man = CombinedEmbeddings([g_man, b_man])

            ds = DataSet("blah")
            ds.read_multiple(train_sets + dev_set + itac_test + conll_test)
            c_man.build_vocabulary([ds])

            if (d, b_dim) not in done_list:
                manager = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                         lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
                # st = time.time()
                manager.train_model_dynamic(train_sets=train_sets, dev_sets=dev_set, max_tries=EPOCH_PATIENCE,
                                            print_interval=10000,
                                            save_path="pointer/models/19_05_11b/glove_d{}_bp_d{}.pt".format(d, b_dim),
                                            teacher_forcing=TEACHER_FORCING)
                # time_list.append(time.time() - st)
            manager2 = PointerManager(g_man, "basic", learning_rate=START_LR, lr_factor=LR_DECAY,
                                      lr_patience=LR_PATIENCE, cuda_device=CUDA_DEVICE)
            manager2.load_model("pointer/models/19_05_11b/glove_d{}_bp_d{}.pt".format(d, b_dim))
            time_list.append(manager2.model.train_time)
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
        result_df["glove dim"] = g_dim_list
        result_df["bpemb dim"] = b_dim_list
        result_df["epochs_trained"] = epoch_list
        result_df["train_time"] = time_list
        result_df["precision_itac"] = itac_p
        result_df["recall_itac"] = itac_r
        result_df["f1_itac"] = itac_f
        result_df["precision_conll"] = conll_p
        result_df["recall_conll"] = conll_r
        result_df["f1_conll"] = conll_f
        result_df.to_csv("results/19_05_11b/ptr_bpemb_glove_search.csv")


def binary_classification_pure_bpemb():
    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier",
                 "BayesianGaussianMixture", "GaussianNB", "LinearSVC", "RandomForest", "GradientBoosting"]
    vs_list = []
    d_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    model_list = []

    for vs in bpemb_vocab_sizes:
        for d in bpemb_dims:
            classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                           SGDClassifier(class_weight="balanced"), BayesianGaussianMixture(), GaussianNB(),
                           LinearSVC(class_weight="balanced"), RandomForestClassifier(class_weight="balanced"),
                           GradientBoostingClassifier()]
            bp_man = BPEmbeddings(bp_vocab_size=vs, dim=d)

            for model, model_name in zip(classifiers, clf_names):
                vs_list.append(vs)
                d_list.append(d)
                model_list.append(model_name)
                clf = BasicClassifier(model=model, emb_man=bp_man, wiki_file=wiki_file)

                st = time.time()
                clf.train_model(data_files=train_sets)
                time_list.append(time.time() - st)

                r, p, f = clf.test_model(itac_test)
                itac_f.append(f)
                itac_p.append(p)
                itac_r.append(r)

                r, p, f = clf.test_model(conll_test)
                conll_f.append(f)
                conll_p.append(p)
                conll_r.append(r)

    result_df = pd.DataFrame()
    result_df["clf_type"] = model_list
    result_df["vocabulary_size"] = vs_list
    result_df["bpemb dim"] = d_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_05_11b/bc_pure_bpemb_grid_search.csv")


def binary_classification_pure_glove():
    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier",
                 "BayesianGaussianMixture", "GaussianNB", "LinearSVC", "RandomForest", "GradientBoosting"]
    g_dim_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    model_list = []

    for path, d in glove_embeddings:

        classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                       SGDClassifier(class_weight="balanced"), BayesianGaussianMixture(), GaussianNB(),
                       LinearSVC(class_weight="balanced"), RandomForestClassifier(class_weight="balanced"),
                       GradientBoostingClassifier()]

        g_man = GloveEmbeddings(path=path, dim=d)

        for model, model_name in zip(classifiers, clf_names):
            g_dim_list.append(d)
            model_list.append(model_name)
            clf = BasicClassifier(model=model, emb_man=g_man, wiki_file=wiki_file)

            st = time.time()
            clf.train_model(data_files=train_sets)
            time_list.append(time.time() - st)

            r, p, f = clf.test_model(itac_test)
            itac_f.append(f)
            itac_p.append(p)
            itac_r.append(r)

            r, p, f = clf.test_model(conll_test)
            conll_f.append(f)
            conll_p.append(p)
            conll_r.append(r)

    result_df = pd.DataFrame()
    result_df["clf_type"] = model_list
    result_df["glove_dim"] = g_dim_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_05_11b/bc_pure_glove_search.csv")


def binary_classification_bpemb_glove():  # Fix Vocabulary Size to 50000
    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier",
                 "BayesianGaussianMixture", "GaussianNB", "LinearSVC", "RandomForest", "GradientBoosting"]
    g_dim_list = []
    b_dim_list = []
    time_list = []
    itac_p = []
    itac_r = []
    itac_f = []
    conll_r = []
    conll_p = []
    conll_f = []
    model_list = []

    for path, d in glove_embeddings:
        for b_dim in bpemb_dims:

            classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                           SGDClassifier(class_weight="balanced"), BayesianGaussianMixture(), GaussianNB(),
                           LinearSVC(class_weight="balanced"), RandomForestClassifier(class_weight="balanced"),
                           GradientBoostingClassifier()]

            g_man = GloveEmbeddings(path=path, dim=d)
            b_man = BPEmbeddings(dim=b_dim, bp_vocab_size=50000)
            c_man = CombinedEmbeddings([g_man, b_man])

            for model, model_name in zip(classifiers, clf_names):
                g_dim_list.append(d)
                b_dim_list.append(b_dim)
                model_list.append(model_name)
                clf = BasicClassifier(model=model, emb_man=c_man, wiki_file=wiki_file)

                st = time.time()
                clf.train_model(data_files=train_sets)
                time_list.append(time.time() - st)

                r, p, f = clf.test_model(itac_test)
                itac_f.append(f)
                itac_p.append(p)
                itac_r.append(r)

                r, p, f = clf.test_model(conll_test)
                conll_f.append(f)
                conll_p.append(p)
                conll_r.append(r)

    result_df = pd.DataFrame()
    result_df["clf_type"] = model_list
    result_df["glove_dim"] = g_dim_list
    result_df["bpemb dim"] = b_dim_list
    result_df["train_time"] = time_list
    result_df["precision_itac"] = itac_p
    result_df["recall_itac"] = itac_r
    result_df["f1_itac"] = itac_f
    result_df["precision_conll"] = conll_p
    result_df["recall_conll"] = conll_r
    result_df["f1_conll"] = conll_f
    result_df.to_csv("results/19_05_11b/bc_bpemb_glove_search.csv")


if __name__ == "__main__":
    # binary_classification_pure_bpemb()
    # binary_classification_pure_glove()
    # binary_classification_bpemb_glove()

    # pointer_bpemb_glove()
    # pointer_with_pure_bpemb()
    pointer_with_pure_glove()

    # create_pointer_examples()
