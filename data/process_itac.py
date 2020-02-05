from dataset import Document, DataSet
from faker import Faker
from faker.providers import internet, date_time
import random


def process_file(source_path, target_path):

    ds = DataSet(target_path)
    with open(source_path, encoding="utf8") as f:
        cur_document = Document()

        for line in f.readlines():
            if line.split()[-1] != "~":
                cur_sentence = []
                cur_anno = []
                anon = 0
                for w in line.split()[1:]:
                    if w == "<ANON>":
                        anon = 1
                    elif w == "</ANON>":
                        anon = 0
                    else:
                        cur_sentence.append(w)
                        cur_anno.append(anon)
                cur_document.sentences.append(list(cur_sentence))
                cur_document.annotated.append(list(cur_anno))

            elif cur_document.sentences:
                ds.add_document(cur_document)
                cur_document = Document()

        if cur_document.sentences:
            ds.add_document(cur_document)

    #  ds.write_data()
    print(ds.word_count, ds.annotation_count)
    return ds


def replace_stuff(ds):
    year_token = "&YEAR"
    mail_token = "&EMAIL"
    date_token = "&DATE"
    time_token = "&TIME"

    fake = Faker()
    fake.add_provider(internet)
    fake.add_provider(date_time)

    c = 0

    for doc in ds.documents:
        for sent, anno in zip(doc.sentences, doc.annotated):
            for index, word in enumerate(sent):
                if word == year_token:
                    sent[index] = fake.year()
                    anno[index] = 1
                    c += 1
                elif word == mail_token:
                    anno[index] = 1
                    c += 1
                    if random.random() < 0.5:
                        sent[index] = fake.safe_email()
                    else:
                        sent[index] = fake.free_email()
                elif word == date_token:
                    anno[index] = 1
                    c += 1
                    sent[index] = str(fake.date_between(start_date="-30y", end_date="today"))
                elif word == time_token:
                    anno[index] = 1
                    c += 1
                    sent[index] = str(fake.time())
    print("Replaced {} tokens".format(c))
    return ds


if __name__ == "__main__":
    ds1 = process_file("raw/itac_data/DEV.bla", "standardized/itac_dev.txt")
    ds1 = replace_stuff(ds1)
    ds1.write_data()

    # This part of the dataset contains no annotations
    # For my work, I manually annotated a small part of this dataset
    ds1 = process_file("raw/itac_data/TRN.txt", "standardized/itac_train0.txt")
    ds1 = replace_stuff(ds1)
    ds1.write_data()

    ds1 = process_file("raw/itac_data/TST.bla", "standardized/itac_test.txt")
    ds1 = replace_stuff(ds1)
    ds1.write_data()
