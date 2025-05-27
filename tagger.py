from abc import abstractmethod
from typing import override

from spacy.lang.zh import Chinese


class Tagger:
    def __init__(self, gazetteer: dict = {}, threshold: int = 1, entity_type_distribution: dict = {}):
        self.gazetteer = gazetteer
        self.threshold = threshold
        for text in entity_type_distribution:
            total_freq = sum(entity_type_distribution[text].values())
            if total_freq >= threshold:  # only keep entities whose occurrence >= threshold
                tp = entity_type_distribution[text].most_common(1)[0][0]
                self.gazetteer[text] = tp

    @abstractmethod
    def predict(self, sentence: str) -> list:
        pass


class RuleBasedTagger(Tagger):
    def __init__(self, gazetteer: dict = {}, threshold: int = 1, entity_type_distribution: dict = {}):
        super().__init__(gazetteer, threshold, entity_type_distribution)
        nlp = Chinese()
        ruler = nlp.add_pipe('entity_ruler')
        patterns = [{'label': label, 'pattern': text}
                    for text, label in gazetteer.items()]
        with nlp.select_pipes(enable='tagger'):
            ruler.add_patterns(patterns)
        self.nlp = nlp

    @override
    def predict(self, sentence: str) -> list:
        doc = self.nlp(sentence)
        entities = []
        for span in doc.ents:
            entities.append({
                'text': span.text,
                'label': span.label_,
                'start': span.start,
                'end': span.end
            })
        return entities
