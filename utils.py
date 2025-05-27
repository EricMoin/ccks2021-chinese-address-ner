from collections import Counter


def get_entity_type_distribution(dataset: list[dict]):
    def get_entities(tokens: list[str], labels: list[str]):
        entities = []
        cur_entity = {}
        for token, label in zip(tokens, labels):
            if label[0] in 'OBS' and cur_entity:
                entities.append(cur_entity)
                cur_entity = {}
            if label[0] in 'BS':
                cur_entity = {
                    'text': token,
                    'type': label[2:]
                }
            elif label[0] in 'IE':
                cur_entity['text'] += token
        if cur_entity:
            entities.append(cur_entity)
        return entities
    entity_type_distribution = {}
    for example in dataset:
        entities = get_entities(example['tokens'], example['labels'])
        for e in entities:
            if '0' in e['text']:  # filter out digits
                continue
            if len(e['text']) <= 2:
                continue
            if e['text'] not in entity_type_distribution:
                entity_type_distribution[e['text']] = Counter()
            # keep only informing types
            if e['type'] in ['prov', 'city', 'district', 'town', 'community', 'devzone', 'village_group', 'road', 'poi']:
                entity_type_distribution[e['text']].update([e['type']])
