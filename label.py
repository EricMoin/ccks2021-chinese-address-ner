from typing import Literal


LabelType = Literal['BIO', 'BIOES']


class LabelMap:
    labels: list[str]
    id2label: dict[int, str]
    label2id: dict[str, int]
    type: LabelType
    entity_types: list[str]

    def __init__(self, labels: list[str], type: LabelType):
        self.labels = []
        self.entity_types = []
        for label in labels:
            self.entity_types.append(label)
            self.labels.append(f"B-{label}")
            self.labels.append(f"I-{label}")
            self.labels.append(f"E-{label}")
            self.labels.append(f"S-{label}")
        self.labels.append('O')
        self.entity_types.append('O')
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.type = type
