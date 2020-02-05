import json
import requests
from dataset import DataSet, Document
from evaluation.metrics import MetricHelper


class NERManager:

    def __init__(self, host, port=9000, timeout=60*1000):
        self.nlp = SNLPHandler(host=host, port=port, timeout=timeout)

    def annotate_doc(self, doc):
        """
        Annotates
        :param Document doc:
        :return:
        """
        predicted_annotations = []

        for sent in doc.sentences:
            text = " ".join(sent)
            c_to_w_dict = self._get_char_to_word_index(sent)
            nlp_anno = self.nlp.annotate(text)
            sent_anno = [0] * len(sent)
            entities = nlp_anno["sentences"][0]["entitymentions"]
            for ent in entities:
                start_w = c_to_w_dict[ent["characterOffsetBegin"]]
                end_w = c_to_w_dict[ent["characterOffsetEnd"]-1]
                for i in range(start_w, end_w+1):
                    try:
                        sent_anno[i] = 1
                    except:
                        print(text)
            predicted_annotations.append(sent_anno)
        return predicted_annotations

    def _get_char_to_word_index(self, sent):
        last_ind = 0
        return_dict = {}
        for i in range(len(sent)):
            for c in range(len(sent[i])):
                return_dict[last_ind] = i
                last_ind += 1
            last_ind += 1
        return return_dict

    def test_ner(self, datasets):
        ds = DataSet(datasets[0])
        ds.read_multiple(datasets)
        predicted_list = []

        for doc in ds.documents:
            predicted = self.annotate_doc(doc)
            predicted_list += predicted
        mh = MetricHelper(predicted=predicted_list, target=ds.get_merged_annotations())
        print("Recall: {}, Precision: {}, F1: {}".format(mh.recall(), mh.precision(), mh.f1()))


class SNLPHandler:

    def __init__(self, host, port=9000, timeout=60*1000):
        self.snlp = StanfordCoreNLP(host + ":" + str(port), server_timeout=int(timeout / 1000))
        self.timeout = timeout

    def annotate(self, text, max_len=70):
        """
        passes text to StanfordCoreNLP Server for processing
        :param str host: host of the server
        :param int port: port number of the server
        :param int timeout: timeout to the server in ms
        :param int max_len: maximum sentence length to be considered during parsing
        :return: dict containing all nlp data
        """

        snlp_data = self.snlp.annotate(text, properties={
                'annotators': 'tokenize,ssplit,pos,lemma,ner',
                'outputFormat': 'json', 'openie.triple.strict': 'true',
                'parse.maxlen': str(max_len), 'ner.maxlen': str(max_len), 'pos.maxlen': str(max_len),
                'ssplit.newlineIsSentenceBreak': 'always',
                'splitter.disable': 'true', 'timeout': self.timeout
                })
        if type(snlp_data) is str:
            raise Exception(snlp_data)

        return snlp_data


class StanfordCoreNLP:

    def __init__(self, server_url, server_timeout=60):
        assert isinstance(server_timeout, int)
        if server_url[-1] == '/':
            server_url = server_url[:-1]
        self.server_url = server_url
        self.server_timeout = server_timeout

    def annotate(self, text, properties=None):
        assert isinstance(text, str)
        if properties is None:
            properties = {}
        else:
            assert isinstance(properties, dict)

        # Checks that the Stanford CoreNLP server is started.
        try:
            requests.get(self.server_url)
        except requests.exceptions.ConnectionError:
            raise Exception('Check whether you have started the CoreNLP server e.g.\n'
                            '$ cd stanford-corenlp-full-2015-12-09/ \n'
                            '$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer')

        data = text.encode()
        r = requests.post(
            self.server_url, params={
                'properties': str(properties)
            }, data=data, headers={'Connection': 'close'}, timeout=self.server_timeout)
        nlp_output = r.text
        if 'outputFormat' in properties and properties['outputFormat'] == 'json':
            try:
                nlp_output = json.loads(nlp_output, encoding='utf-8', strict=True)
            except:
                pass
        return nlp_output


if __name__ == "__main__":
    # This requires starting a StanfordCoreNLP instance either locally or on a remote machine
    ner_man = NERManager("http://localhost")

    print("\nconll_train:")
    ner_man.test_ner(["../data/standardized/conll_train.txt"])
    print("\nconll_valid:")
    #ner_man.test_ner(["../data/standardized/conll_valid.txt"])
    print("\nconll_test:")
    #ner_man.test_ner(["../data/standardized/conll_test.txt"])
    print("\nitac_dev:")
    #ner_man.test_ner(["../data/standardized/itac_dev.txt"])
    print("\nitac_test:")
    #ner_man.test_ner(["../data/standardized/itac_test.txt"])
