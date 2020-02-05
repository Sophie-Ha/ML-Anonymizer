from embeddings.embedding_manager import BPEmbeddings, GloveEmbeddings, CombinedEmbeddings
from pointer.ptr_manager import PointerManager
from pointer.ptr_flair_manager import FlairPointerManager
from binary_classification.basic_classifier import BasicClassifier
from dataset import DataSet
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, Embeddings
from sklearn.ensemble import RandomForestClassifier

START_LR = 0.01
LR_PATIENCE = 3
LR_DECAY = 0.5
EPOCH_PATIENCE = 6
MAX_EPOCHS = 20
TEACHER_FORCING = 0.5


if __name__ == '__main__':

    embeddings = StackedEmbeddings([FlairEmbeddings('news-forward-fast'), FlairEmbeddings('news-backward-fast')])
    model_file = "pointer/models/19_06_06/flair_itac+conll.pt"

    man = FlairPointerManager(embeddings, learning_rate=START_LR, lr_factor=LR_DECAY, lr_patience=LR_PATIENCE,
                              n_encoder_layers=1, n_decoder_layers=1, cuda_device=0)
    man.load_model(model_file)
    man.test_misclassifications(["data/standardized/itac_test.txt"],
                                "results/19_06_06/flair_ptr_misclassifications_itac.csv")
    man.test_misclassifications(["data/standardized/conll_test.txt"],
                                "results/19_06_06/flair_ptr_misclassifications_conll.csv")

    print("Testing and training Random Forest")

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    model = BasicClassifier(clf, embeddings, emb_type="flair")
    model.train_model(data_files=["data/standardized/conll_train.txt", "data/standardized/itac_dev.txt",
                                  "data/standardized/itac_train0.txt"])
    model.test_misclassifications(["data/standardized/itac_test.txt"],
                                  "results/19_06_06/flair_bc_misclassifications_itac.csv")
    model.test_misclassifications(["data/standardized/conll_test.txt"],
                                  "results/19_06_06/flair_bc_misclassifications_conll.csv")

