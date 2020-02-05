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
from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet

train_sets = ["data/standardized/conll_train.txt", "data/standardized/itac_dev.txt", "data/standardized/itac_train0.txt"]
dev_set = ["data/standardized/conll_valid.txt"]
itac_test = ["data/standardized/itac_test.txt"]
conll_test = ["data/standardized/conll_test.txt"]
rsics_test = ["data/standardized/rsics_test.txt"]

wiki_file = "embeddings/enwiki-20190320-words-frequency.txt"


def bc_test_case_sensitivity():
    embeddings = StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')])

    model_list = []
    case_list = []
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

    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier", "LinearSVC", "RandomForest",
                 "GradientBoosting"]
    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                   RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

    for clf, clf_name in zip(classifiers, clf_names):
        model_list.append(clf_name)
        case_list.append("True")

        model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file, lowercase=True)

        st = time.time()
        model.train_model(data_files=train_sets + dev_set)
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
        result_df["lowercase"] = case_list
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
        result_df.to_csv("results/19_07_02/bc_case_sensitivity.csv")

    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                   RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

    for clf, clf_name in zip(classifiers, clf_names):
        model_list.append(clf_name)
        case_list.append("False")

        model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file, lowercase=False)

        st = time.time()
        model.train_model(data_files=train_sets + dev_set)
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
        result_df["lowercase"] = case_list
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
        result_df.to_csv("results/19_07_02/bc_case_sensitivity.csv")


def bc_test_sent_embeddings():
    embeddings = WordEmbeddings("en-glove")
    model_list = []
    sent_emb_list = []
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

    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier", "LinearSVC", "RandomForest",
                 "GradientBoosting"]
    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                   RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

    for clf, clf_name in zip(classifiers, clf_names):
        model_list.append(clf_name)
        sent_emb_list.append("True")

        model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file, embed_sentences=True)

        st = time.time()
        model.train_model(data_files=train_sets + dev_set)
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
        result_df["lowercase"] = sent_emb_list
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
        result_df.to_csv("results/19_07_02/bc_sent_embedding.csv")

    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                   RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]

    for clf, clf_name in zip(classifiers, clf_names):
        model_list.append(clf_name)
        sent_emb_list.append("False")

        model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file, embed_sentences=False)

        st = time.time()
        model.train_model(data_files=train_sets + dev_set)
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
        result_df["embed_sentences"] = sent_emb_list
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
        result_df.to_csv("results/19_07_02/bc_sent_embedding.csv")


if __name__ == "__main__":
    bc_test_case_sensitivity()
    bc_test_sent_embeddings()
