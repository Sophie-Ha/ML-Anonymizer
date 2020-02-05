from dataset import Document, DataSet


def process_file(source_path, target_path):

    ds = DataSet(target_path)
    with open(source_path, encoding="utf8") as f:
        cur_document = Document()
        cur_sentence = []
        cur_anno = []

        for line in f.readlines():
            if len(line) > 1:
                temp_word_list = line.split()
                if temp_word_list[0] == "-DOCSTART-":
                    if cur_document.sentences:
                        ds.add_document(cur_document)
                        cur_document = Document()
                else:
                    cur_sentence.append(temp_word_list[0])
                    anno = temp_word_list[-1]
                    if anno[0] == "B":  # marks the first word of a named entity
                        cur_anno.append(1)
                    elif anno[0] == "I":  # marks any word of a named entity except the first
                        cur_anno.append(2)
                    elif temp_word_list[1] == "CD":  # marks dates and numbers
                        cur_anno.append(1)
                    else:
                        cur_anno.append(0)
            elif cur_sentence:
                cur_document.sentences.append(list(cur_sentence))
                cur_document.annotated.append(list(cur_anno))
                cur_sentence = []
                cur_anno = []

        if cur_document.sentences:
            ds.add_document(cur_document)

    ds.write_data()
    print(ds.word_count, ds.annotation_count)


if __name__ == "__main__":
    process_file("raw/conll/test.txt", "standardized/conll_test.txt")
    process_file("raw/conll/train.txt", "standardized/conll_train.txt")
    process_file("raw/conll/valid.txt", "standardized/conll_valid.txt")
