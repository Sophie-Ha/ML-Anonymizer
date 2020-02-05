import pandas as pd
import requests
import time
import os

from textblob import TextBlob
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet, Document

"""
This script evaluates the performance of Sentiment analysis and Keyphrase extraction tasks before and after text has been anonymized.
Since Sentiment analysis and keyphrase extraction are performed using Microsoft Azure, you will need your own API key
"""

ms_api_key = ""  # Paste your API key here
sentiment_url = "https://westeurope.api.cognitive.microsoft.com/text/analytics/v2.1/sentiment"  # might need to change this depending on location
keyphrase_url = "https://westeurope.api.cognitive.microsoft.com/text/analytics/v2.1/keyPhrases"  # might need to change this depending on location

train_sets = ["data/standardized/conll_train.txt", "data/standardized/itac_dev.txt", "data/standardized/itac_train0.txt"]
dev_set = ["data/standardized/conll_valid.txt"]

wiki_file = "embeddings/enwiki-20190320-words-frequency.txt"

keyphrase_directory = "data/keyphrase"

data_sub_path = "data/sentiment/rotten_imdb/quote.tok.gt9.5000"
data_obj_path = "data/sentiment/rotten_imdb/plot.tok.gt9.5000"

polarity_neg_path = "data/sentiment/rt-polaritydata/rt-polarity.neg"
polarity_pos_path = "data/sentiment/rt-polaritydata/rt-polarity.pos"


def test_subjectivity(clf_path, clf, save_path):
    clf_model = get_classifier(clf_path, clf)

    # load and anonymize test data
    ds_obj = prepare_data(clf_model, data_obj_path)
    ds_sub = prepare_data(clf_model, data_sub_path)

    # perform tests on anonymized and unanonymized data
    pure_obj, anon_obj = get_subjectivity_scores(ds_obj)
    pure_sub, anon_sub = get_subjectivity_scores(ds_sub)

    # save results
    obj_df = pd.DataFrame()
    obj_df["pure_objectivity"] = pure_obj
    obj_df["anon_objectivity"] = anon_obj
    obj_df["target_objectivity"] = 0.0

    sub_df = pd.DataFrame()
    sub_df["pure_objectivity"] = pure_sub
    sub_df["anon_objectivity"] = anon_sub
    sub_df["target_objectivity"] = 1.0

    res_df = pd.concat([obj_df, sub_df], ignore_index=True, sort=False)
    res_df.to_csv(save_path)


def test_polarity(clf_path, clf, save_path):
    clf_model = get_classifier(clf_path, clf)

    # load and anonymize test data
    ds_neg = prepare_data(clf_model, polarity_neg_path)
    ds_pos = prepare_data(clf_model, polarity_pos_path)

    # perform tests on anonymized and unanonymized data
    pure_neg, anon_neg = get_polarity_scores(ds_neg)
    pure_pos, anon_pos = get_polarity_scores(ds_pos)

    # save results
    obj_df = pd.DataFrame()
    obj_df["pure_polarity"] = pure_neg
    obj_df["anon_polarity"] = anon_neg
    obj_df["target_polarity"] = -1.0

    sub_df = pd.DataFrame()
    sub_df["pure_polarity"] = pure_pos
    sub_df["anon_polarity"] = pure_neg
    sub_df["target_polarity"] = 1.0

    res_df = pd.concat([obj_df, sub_df], ignore_index=True, sort=False)
    res_df.to_csv(save_path)


def get_classifier(path, clf):
    embeddings = StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')])
    model = BasicClassifier(clf, embeddings, emb_type="flair", wiki_file=wiki_file, lowercase=True)

    try:
        model.load_model(path=path)
    except FileNotFoundError:
        print("No saved model was found for the given path. Training one instead...")
        #model.train_model(data_files=train_sets + dev_set)
        print("Done training")
        #model.save_model(path)

    return model


def test_polarity_azure(clf_path, clf, save_path, anon=True):
    clf_model = get_classifier(clf_path, clf)

    # load and anonymize test data
    ds_neg = prepare_data(clf_model, polarity_neg_path)
    ds_pos = prepare_data(clf_model, polarity_pos_path)

    neg_doc = build_document_json(ds_neg, anon)
    pos_doc = build_document_json(ds_pos, anon)

    neg_scores = get_sentiment_azure(neg_doc)
    pos_scores = get_sentiment_azure(pos_doc)
    targets = [0.0] * len(neg_scores) + [1.0] * len(pos_scores)
    res_df = pd.DataFrame()
    res_df["sent_pred"] = neg_scores + pos_scores
    res_df["targets"] = targets
    res_df.to_csv(save_path)


def test_keyphrase_azure(clf_path, clf, save_path, anon=True):
    clf_model = get_classifier(clf_path, clf)

    document_dicts, target_dicts = prepare_keyphrase_data(clf_model=clf_model, data_dir=keyphrase_directory)
    json_list = build_keyphrase_json(documents=document_dicts, anon=anon)
    predicted_dicts = get_keyphrases_azure(json_list)

    with open(save_path, "w") as f:
        fp, tp, fn = get_keyphrase_scores(target_dict=target_dicts, res_dict=predicted_dicts)
        f.write("Scores on Keyphrase level:\n")
        f.write("FP: {}\nTP: {}\nFN: {}\n".format(fp, tp, fn))
        if tp+fp > 0:
            p = tp/(tp+fp)
        else:
            p = 0
        if tp + fn > 0:
            r = tp/(tp+fn)
        else:
            r = 0
        if p+r > 0:
            f1 = (2*p*r)/(p+r)
        else:
            f1 = 0
        f.write("P: {}\nR: {}\nF1: {}\n".format(p, r, f1))
        print("Scores on Keyphrase level:")
        print("FP: {}\nTP: {}\nFN: {}".format(fp, tp, fn))
        print("P: {}\nR: {}\nF1: {}".format(p, r, f1))

        fp, tp, fn = get_keyphrase_scores_word_level(target_dict=target_dicts, res_dict=predicted_dicts)
        f.write("\nScores on Word Level:\n")
        f.write("FP: {}\nTP: {}\nFN: {}\n".format(fp, tp, fn))
        if tp+fp > 0:
            p = tp/(tp+fp)
        else:
            p = 0
        if tp + fn > 0:
            r = tp/(tp+fn)
        else:
            r = 0
        if p+r > 0:
            f1 = (2*p*r)/(p+r)
        else:
            f1 = 0
        f.write("P: {}\nR: {}\nF1: {}\n".format(p, r, f1))
        print("Scores on Word level:")
        print("FP: {}\nTP: {}\nFN: {}".format(fp, tp, fn))
        print("P: {}\nR: {}\nF1: {}".format(p, r, f1))


def build_document_json(ds, anon):
    d_list_raw = []
    for i, doc in enumerate(ds.documents):
        d_dict = {"id": str(i),
                  "language": "en",
                  "text": doc.create_text(anon=anon)}
        d_list_raw.append(d_dict)

    json_list = []
    n = int(len(d_list_raw)/10)
    for i in range(9):
        json_raw = {"documents": d_list_raw[i*n:(i+1)*n]}
        json_list.append(json_raw)
    json_list.append({"documents": d_list_raw[9*n:]})
    return json_list


def get_keyphrase_scores(target_dict, res_dict):
    """ Checks for exact keyphrase matches """
    # total_count = 0
    fp = 0
    tp = 0
    fn = 0
    for key in target_dict:
        if key in res_dict:
            t_list = target_dict[key]
            p_list = res_dict[key]
            for t in t_list:
                if t in p_list:
                    tp += 1
                else:
                    fn += 1
            for t in p_list:
                if t not in t_list:
                    fp += 1
    return fp, tp, fn


def get_keyphrase_scores_word_level(target_dict, res_dict):
    """ Checks keyphrase results on a word-level """
    fp = 0
    tp = 0
    fn = 0
    for key in target_dict:
        if key in res_dict:
            t_list = []
            for t in [keyphrase.split() for keyphrase in target_dict[key]]:
                t_list += t
            t_list.sort()

            p_list = []
            for p in [keyphrase.split() for keyphrase in res_dict[key]]:
                p_list += p
            p_list.sort()

            for t in t_list:
                if t in p_list:
                    tp += 1
                    p_list.remove(t)
                else:
                    fn += 1
            fp += len(p_list)
    return fp, tp, fn


def build_keyphrase_json(documents, anon):
    d_list_raw = []
    for i in documents:
        d_dict = {"id": i,
                  "language": "en",
                  "text": documents[i].create_text(anon=anon)}
        d_list_raw.append(d_dict)

    json_list = []
    n = int(len(d_list_raw) / 10)
    for i in range(9):
        json_raw = {"documents": d_list_raw[i * n:(i + 1) * n]}
        json_list.append(json_raw)
    json_list.append({"documents": d_list_raw[9 * n:]})
    return json_list


def prepare_data(clf_model, data_path):
    """
    loads and anonymizes test data
    :param BasicClassifier clf_model:
    :param data_path:
    :return: dataset containing test data with anon annotations
    """
    ds = DataSet(path="")

    with open(data_path, encoding="ISO-8859-1") as f:
        for line in f.readlines():
            raw_text = str(line)
            doc = Document(raw_text=raw_text)
            doc.create_from_text()
            doc.annotated = clf_model._annotate_document(doc)
            ds.add_document(document=doc)
    print("Number of annotated words:", ds.annotation_count)
    return ds


def prepare_keyphrase_data(clf_model, data_dir):
    all_data_files = os.listdir(data_dir)
    main_data_files = list(set([datafile.split(".")[-2] for datafile in all_data_files]))

    documents = {}
    targets = {}

    for f_name in main_data_files:
        with open(data_dir + "/" + f_name + ".txt", encoding="ISO-8859-1") as f:
            # TODO maybe replace " with \"
            raw_text = ". ".join([str(line).strip() for line in f.readlines()])
            doc = Document(raw_text=raw_text)
            doc.create_from_text()
            doc.annotated = clf_model._annotate_document(doc)
            documents[f_name] = doc
        with open(data_dir + "/" + f_name + ".key", encoding="ISO-8859-1") as f:
            t_list = []
            for line in f.readlines():
                t_list.append(str(line).strip())
            targets[f_name] = t_list

    return documents, targets


def get_keyphrases_azure(json_list):
    headers = {"Ocp-Apim-Subscription-Key": ms_api_key}
    res_dict = {}
    for d_json in json_list:
        print("Posting Request")
        t = time.time()
        response = requests.post(keyphrase_url, headers=headers, json=d_json)
        print("finished. took {} seconds".format(time.time() - t))
        keyphrases = response.json()
        if "documents" not in keyphrases:
            print(keyphrases)
        if "errors" in keyphrases:
            for e in keyphrases["errors"]:
                print(e)
        for doc_dict in keyphrases["documents"]:
            res_dict[doc_dict["id"]] = doc_dict["keyPhrases"]
    return res_dict


def get_sentiment_azure(json_list):
    headers = {"Ocp-Apim-Subscription-Key": ms_api_key}
    score_list = []
    for d_json in json_list:
        print("Posting Request")
        t = time.time()
        response = requests.post(sentiment_url, headers=headers, json=d_json)
        print("finished. took {} seconds".format(time.time() - t))
        sentiments = response.json()
        if "documents" not in sentiments:
            print(sentiments)
        if "errors" in sentiments:
            for e in sentiments["errors"]:
                print(e)
        for doc_dict in sentiments["documents"]:
            score_list.append(doc_dict["score"])
    return score_list


def get_polarity_scores(ds):
    pure_scores = []
    anon_scores = []

    for doc in ds.documents:
        raw_text = doc.create_text(anon=False)
        blob = TextBlob(raw_text)
        pure_scores.append(blob.sentiment.polarity)

        anon_text = doc.create_text(anon=True)
        blob = TextBlob(anon_text)
        anon_scores.append(blob.sentiment.polarity)

    return pure_scores, anon_scores


def get_subjectivity_scores(ds):
    pure_scores = []
    anon_scores = []

    for doc in ds.documents:
        raw_text = doc.create_text(anon=False)
        blob = TextBlob(raw_text)
        pure_scores.append(blob.sentiment.subjectivity)

        anon_text = doc.create_text(anon=True)
        blob = TextBlob(anon_text)
        anon_scores.append(blob.sentiment.subjectivity)

    return pure_scores, anon_scores


if __name__ == "__main__":

    clf_names = ["AdaBoostClassifier", "LogisticRegression", "SGDClassifier", "LinearSVC", "RandomForest",
                 "GradientBoosting"]
    classifiers = [AdaBoostClassifier(), LogisticRegression(class_weight="balanced"),
                   SGDClassifier(class_weight="balanced"), LinearSVC(class_weight="balanced"),
                   RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()]
    """

    test_polarity_azure(clf_path="binary_classification/models/" + clf_names[0] + "_default_flair", clf=classifiers[0],
                        save_path="results/sentiment/azure_baseline_polarity.csv", anon=False)

    #for clf, clf_name in zip(classifiers, clf_names):
    #    print("\nWorking on " + clf_name + "\n")
    #    test_polarity_azure(clf_path="binary_classification/models/" + clf_name + "_default_flair", clf=clf,
    #                        save_path="results/sentiment/azure_" + clf_name + "_polarity.csv")

    """
    """
    ds = DataSet("data/standardized/conll_valid.txt")
    ds.read_data()
    print("Dataset has {} words, {} sentences, and {} documents\n".format(ds.word_count, ds.sentence_count,
                                                                          len(ds.documents)))

    for clf_name, clf in zip(clf_names, classifiers):
        model = get_classifier(path="binary_classification/models/" + clf_name + "_default_flair", clf=clf)
        print(clf_name)
        t = time.time()
        for doc in ds.documents:
            model._annotate_document(doc)
        print("time to annotate: {}\n".format(time.time()-t))
    """

    # test_keyphrase_azure(clf_path="binary_classification/models/" + clf_names[0] + "_default_flair", clf=classifiers[0],
    #                      save_path="results/keyphrase/azure_baseline_keyphrase.csv", anon=False)

    for clf, clf_name in zip(classifiers, clf_names):
        print("\nWorking on " + clf_name + "\n")
        test_keyphrase_azure(clf_path="binary_classification/models/" + clf_name + "_default_flair", clf=clf,
                             save_path="results/keyphrase/azure_" + clf_name + "_keyphrase.csv")
