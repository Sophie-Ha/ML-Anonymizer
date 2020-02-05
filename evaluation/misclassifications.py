from dataset import Document, DataSet
import pandas as pd


def analyze_misclassifications(ds):
    """

    :param DataSet ds: Dataset with annotations and predictions
    :return: pd.DataFrame that contains all sentences with misclassifications
    """

    def build_string_anonymized(sent, anon_list):
        word_list = []
        for w, a in zip(sent, anon_list):
            if a != 0:
                word_list.append("<ANON>")
            else:
                word_list.append(w)
        return " ".join(word_list)

    assert len(ds.documents[0].annotated) == len(ds.documents[0].predicted) == len(ds.documents[0].sentences), \
        "Dataset needs to contain annotations and predictions"

    raw_text_list = []
    annotated_list = []
    predicted_list = []

    for doc in ds.documents:
        for sent, anno, pred in zip(doc.sentences, doc.annotated, doc.predicted):
            if anno != list(pred):
                raw_text_list.append(" ".join(sent))
                annotated_list.append(build_string_anonymized(sent, anno))
                predicted_list.append(build_string_anonymized(sent, list(pred)))

    result_df = pd.DataFrame()
    result_df["raw_text"] = raw_text_list
    result_df["target_anon"] = annotated_list
    result_df["predicted_anon"] = predicted_list
    return result_df
