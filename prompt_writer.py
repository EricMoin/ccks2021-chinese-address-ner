from tagger import Tagger


class PromptWriter:
    def __init__(self, conll_file: str, tagger: Tagger, eng2chn: dict = None):
        self.conll_file = conll_file
        self.tagger = tagger
        self.eng2chn = {
            'prov': '省',
            'city': '市',
            'district': '区县',
            'town': '镇街道',
            'community': '社区村',
            'devzone': '开发区',
            'village_group': '村',
            'road': '路',
            'poi': '兴趣点'
        }
        if eng2chn is not None:
            self.eng2chn.update(eng2chn)

    def write(self, sentence: list[dict]) -> str:
        with open(self.conll_file, 'w', encoding='utf8') as fout:
            for data in sentence:
                for token, label in zip(data['tokens'], data['labels']):
                    print(token, label, sep=' ', file=fout)
                tagged_entities = self.tagger.predict(''.join(data['tokens']))
                if len(tagged_entities) > 0:
                    print('<EOS>', 'X', sep=' ', file=fout)
                    prompt = ''
                    for e in tagged_entities:
                        prompt += e['text'] + '是' + \
                            self.eng2chn[e['label']] + '，'
                    for token in prompt:
                        print(token, 'X', sep=' ', file=fout)
                print('', file=fout)

    def write_test(self, test_file: str):
        with open(test_file, 'r', encoding='utf8') as fin, \
                open(self.conll_file, 'w', encoding='utf8') as fout:
            for line in fin:
                _, text = line.strip().split('\u0001')
                for token in text:
                    print(token, 'O', sep=' ', file=fout)
                tagged_entities = self.tagger.predict(text)
                if len(tagged_entities) > 0:
                    print('<EOS>', 'X', sep=' ', file=fout)
                    prompt = ''
                    for e in tagged_entities:
                        prompt += e['text'] + '是' + \
                            self.eng2chn[e['label']] + '，'
                    for token in prompt:
                        print(token, 'X', sep=' ', file=fout)
                print('', file=fout)
