from pointer.ptr_flair_manager import FlairPointerManager
from dataset import DataSet, Document
from flair.embeddings import WordEmbeddings
import time



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
