from dataset import DataSet, Document
from pointer.flair_ptr import FlairPointer
from evaluation.metrics import MetricHelper
from evaluation.misclassifications import analyze_misclassifications
import torch
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, Embeddings
from flair.data import Sentence
import datetime
import time


class FlairPointerManager:

    def __init__(self, embeddings, learning_rate=0.1,
                 lr_factor=None, lr_patience=None, n_encoder_layers=1, n_decoder_layers=1, cuda_device=0):
        """

        :param Embeddings embeddings:
        :param learning_rate:
        :param lr_factor:
        :param lr_patience:
        :param n_encoder_layers:
        :param n_decoder_layers:
        """
        self.device = torch.device("cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu")
        self.embeddings = embeddings
        self.embeddings.to(self.device)

        self.model = FlairPointer(hidden_size=self.embeddings.embedding_length,
                                  embedding_size=self.embeddings.embedding_length,
                                  learning_rate=learning_rate, lr_factor=lr_factor, lr_patience=lr_patience,
                                  n_encoder_layers=n_encoder_layers, n_decoder_layers=n_decoder_layers)

    def train_model(self, datasets, n_epochs=5, print_interval=500, save_path="", teacher_forcing=0.5, dev_sets=None):
        input_list, target_list, _ = self.make_training_data(datasets)
        if dev_sets:
            dev_inputs, dev_targets, _ = self.make_training_data(dev_sets)
        else:
            dev_inputs = None
            dev_targets = None
        print("Prepared Training data")
        print("Training")
        self.model.train(inputs=input_list, targets=target_list, epochs=n_epochs, print_interval=print_interval,
                         teacher_forcing=teacher_forcing, dev_inputs=dev_inputs, dev_targets=dev_targets)

        if save_path:
            self.model.save_model_parameters(save_path)

    def train_model_dynamic(self, train_sets, dev_sets, max_tries=5, print_interval=500, save_path="",
                            teacher_forcing=0.5, max_epochs=20):
        train_input, train_target, _ = self.make_training_data(train_sets)
        dev_input, dev_target, dev_ds = self.make_training_data(dev_sets)
        print("Prepared Data")

        best_f1 = 0
        n_tries = 0
        n_epochs = 0
        best_epoch = 0

        while n_tries < max_tries and n_epochs < max_epochs:
            n_epochs += 1
            #  train model
            self.model.train(inputs=train_input, targets=train_target, epochs=1, print_interval=print_interval,
                             teacher_forcing=teacher_forcing, dev_inputs=dev_input, dev_targets=dev_target)

            #  test model
            with torch.no_grad():
                predicted = []
                for input_sent in dev_input:
                    res_indices = self.model.predict(input_sent)
                    cur_predicted = []
                    for i in range(len(input_sent) - 1):
                        if i in res_indices:
                            cur_predicted.append(1)
                        else:
                            cur_predicted.append(0)
                    predicted.append(cur_predicted)
                mh = MetricHelper(predicted=predicted, target=dev_ds.get_merged_annotations())
                cur_f1 = mh.f1()

            if cur_f1 >= best_f1:
                best_f1 = cur_f1
                n_tries = 0
                self.model.save_model_parameters(save_path)
                print("New best model with f1={} saved\n".format(cur_f1))
                best_epoch = n_epochs
            else:
                n_tries += 1
                print("f1={}\n".format(cur_f1))
            print("{} epochs remaining".format(max_epochs-n_epochs))
        print("Finished training after {} epochs, {} seconds. Best model achieved after {} epochs".format(n_epochs,
                                                                                                          self.model.train_time,
                                                                                                          best_epoch))

    def test_model(self, datasets):
        input_list, target_list, ds = self.make_training_data(datasets)
        print("Prepared Test data")
        predicted = []
        with torch.no_grad():
            for input_sent in input_list:
                res_indices = self.model.predict(input_sent)
                cur_predicted = []
                for i in range(len(input_sent)-1):
                    if i in res_indices:
                        cur_predicted.append(1)
                    else:
                        cur_predicted.append(0)
                predicted.append(cur_predicted)
        mh = MetricHelper(predicted=predicted, target=ds.get_merged_annotations())
        print("Recall: {}, Precision: {}, F1: {}".format(mh.recall(), mh.precision(), mh.f1()))
        return mh.recall(), mh.precision(), mh.f1()

    def annotate_string(self, input_text):
        doc = Document(raw_text=input_text)
        doc.create_from_text()
        annotated = self._annotate_document(doc)
        doc.annotated = annotated
        return doc.create_text()

    def _annotate_document(self, doc):
        input_list = []
        for sent in doc.sentences:
            flair_sent = Sentence(" ".join(sent))
            self.embeddings.embed(flair_sent)
            input_list.append(flair_sent)

        predicted = []
        for input_sent in input_list:
            res_indices = self.model.predict(input_sent)
            cur_predicted = []
            for i in range(len(input_sent)):
                if i in res_indices:
                    cur_predicted.append(1)
                else:
                    cur_predicted.append(0)
            predicted.append(cur_predicted)
        return predicted

    def test_misclassifications(self, datasets, csv_path):
        ds = DataSet(datasets[0], reduce_to_one=True)
        ds.read_multiple(datasets)

        for doc in ds.documents:
            doc.predicted = self._annotate_document(doc)

        df = analyze_misclassifications(ds)
        df.to_csv(csv_path)

    def load_model(self, path):
        self.model.load_model_parameters(path)

    def make_training_data(self, datasets, red_to_one=True):
        ds = DataSet(datasets[0], reduce_to_one=red_to_one)
        ds.read_multiple(datasets)

        input_list = []
        target_list = []

        for doc in ds.documents:
            for sent, anno in zip(doc.sentences, doc.annotated):
                flair_sent = Sentence(" ".join(sent))
                self.embeddings.embed(flair_sent)
                input_list.append(flair_sent)
                trgt = [i for i in range(len(anno)) if anno[i] == 1]
                trgt.append(len(anno))
                target_list.append(trgt)
        return input_list, target_list, ds


def read_model_info(path):
    print("Loading model from", path)
    checkpoint = torch.load(path)
    print("Model was saved on {} and trained for {} with parameters:".format(checkpoint["datetime"],
                                                                             datetime.timedelta(seconds=checkpoint["train_time"])))
    params = ["lr_scheduling", "learning_rate", "lr_factor", "lr_patience", "epoch", "sos_token"]
    for param in params:
        if param in checkpoint:
            print(param + ": " + str(checkpoint[param]))
        else:
            print(param, "was not saved.")


if __name__ == '__main__':
    START_LR = 0.01
    LR_PATIENCE = 3
    LR_DECAY = 0.5
    EPOCH_PATIENCE = 6
    MAX_EPOCHS = 2
    TEACHER_FORCING = 0.5

    fasttext = WordEmbeddings("en-crawl")

    man2 = FlairPointerManager(fasttext, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                               n_encoder_layers=1, n_decoder_layers=1)
    # man2.load_model("models/flair_test.pt")

    ds = DataSet("data/standardized/conll_valid.txt")
    ds.read_data()
    for _ in range(5):
        t = time.time()
        for doc in ds.documents:
            man2._annotate_document(doc)
        print("time to annotate: {}\n".format(time.time() - t))
