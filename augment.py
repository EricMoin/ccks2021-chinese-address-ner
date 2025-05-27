import collections
import os
from conll_reader import ConllReader, ConllEntity
from label import LabelMap
import yaml  # type: ignore
import random
import torch  # Added for torch.softmax

from predictor import Predictor  # Added
# Added, aliased to avoid conflict if any
from config import AugmentConfig  # MODIFIED: Import AugmentConfig


def get_entity_type_from_label(label: str) -> str:
    """Converts a BIOES label to its core entity type (e.g., B-prov -> prov)."""
    if '-' in label:
        return label.split('-', 1)[1]
    return label  # For 'O' tag or other non-BIOES tags


def extract_entity_type_patterns(conll_file: str, min_freq: int = 5) -> collections.Counter[tuple[str, ...]]:
    """
    Extracts frequent entity type patterns from a CoNLL file.
    A pattern is a tuple of entity types in a sentence.
    e.g., ('prov', 'prov', 'O', 'city', 'city')
    """
    reader = ConllReader()
    type_patterns = collections.Counter()
    for conll_entity in reader.read(conll_file):
        current_pattern_types = []
        for label in conll_entity.labels:
            entity_type = get_entity_type_from_label(label)
            current_pattern_types.append(entity_type)
        type_patterns[tuple(current_pattern_types)] += 1

    # Filter for patterns with minimum frequency
    frequent_patterns = collections.Counter(
        {pattern: count for pattern, count in type_patterns.items() if count >= min_freq})
    return frequent_patterns


def extract_entities_by_type(conll_file: str) -> dict[str, list[list[str]]]:
    """
    Extracts entities from a CoNLL file and groups them by their core type.
    Returns a dictionary like: {'prov': [['北', '京', '市']], 'city': [['朝', '阳', '区']]}
    Each entity is stored as a list of tokens.
    """
    reader = ConllReader()
    entities_by_type = collections.defaultdict(list)
    for conll_entity in reader.read(conll_file):
        current_entity_tokens: list[str] = []
        current_entity_type: str | None = None
        for token, label in zip(conll_entity.tokens, conll_entity.labels):
            entity_type = get_entity_type_from_label(label)
            # Default to 'O' if no prefix
            bio_prefix = label.split('-', 1)[0] if '-' in label else 'O'

            if bio_prefix == 'B':  # Start of a new entity
                if current_entity_tokens and current_entity_type:  # Save previous entity
                    entities_by_type[current_entity_type].append(
                        list(current_entity_tokens))
                current_entity_tokens = [token]
                current_entity_type = entity_type
            elif bio_prefix == 'S':  # Single token entity
                if current_entity_tokens and current_entity_type:  # Save previous entity, if any
                    entities_by_type[current_entity_type].append(
                        list(current_entity_tokens))
                entities_by_type[entity_type].append([token])
                current_entity_tokens = []
                current_entity_type = None
            # Continuation or end of current entity
            elif bio_prefix in ['I', 'E'] and current_entity_type == entity_type:
                current_entity_tokens.append(token)
                if bio_prefix == 'E':
                    if current_entity_tokens and current_entity_type:
                        entities_by_type[current_entity_type].append(
                            list(current_entity_tokens))
                    current_entity_tokens = []
                    current_entity_type = None
            # 'O' tag or unexpected transition (e.g. B-city followed by B-prov)
            else:
                if current_entity_tokens and current_entity_type:  # Save previous entity
                    entities_by_type[current_entity_type].append(
                        list(current_entity_tokens))
                current_entity_tokens = []
                current_entity_type = None
                # If it's an O tag, nothing more to do for this token regarding entity accumulation
                # If it was an unexpected transition (like B-city then B-prov), the next iteration's 'B' or 'S' will handle it.

        # Save any trailing entity
        if current_entity_tokens and current_entity_type:
            entities_by_type[current_entity_type].append(
                list(current_entity_tokens))

    # Deduplicate lists of tokens for each entity type
    for entity_type in entities_by_type:
        # Convert list of lists to list of tuples for hashing, then back to list of lists
        unique_entities = sorted(
            list(set(map(tuple, entities_by_type[entity_type]))))
        entities_by_type[entity_type] = [
            list(entity) for entity in unique_entities]

    return entities_by_type


def generate_bioes_labels(tokens: list[str], entity_type: str) -> list[str]:
    """Generates BIOES labels for a list of tokens given an entity type."""
    if not tokens:
        return []
    if len(tokens) == 1:
        return [f"S-{entity_type}"]
    labels = [f"B-{entity_type}"]
    for _ in range(1, len(tokens) - 1):
        labels.append(f"I-{entity_type}")
    labels.append(f"E-{entity_type}")
    return labels


def augment_data_by_replacement(
    original_conll_file: str,
    frequent_patterns: collections.Counter[tuple[str, ...]],
    entities_by_type: dict[str, list[list[str]]],
    # Added for consistency, though not strictly used in this func yet
    label_map_instance: LabelMap,
    # How many new sentences to try generate per original sentence
    augmentation_factor: int = 1,
    max_new_sentences: int = 10000  # Global cap on augmented sentences
) -> list[ConllEntity]:
    """
    Augments data by replacing entities in sentences that match frequent patterns.
    """
    reader = ConllReader()
    augmented_sentences: list[ConllEntity] = []
    original_sentences_processed = 0

    # Create a copy of entities_by_type to allow safe removal of used entities if needed for variety
    available_entities = collections.defaultdict(list)
    for type_key, ent_list in entities_by_type.items():
        available_entities[type_key] = [list(e) for e in ent_list]  # Deep copy

    source_entities: list[ConllEntity]
    if isinstance(original_conll_file, str):
        source_entities = list(reader.read(original_conll_file))
    elif isinstance(original_conll_file, list):
        source_entities = original_conll_file
    else:
        raise TypeError(
            "original_conll_file must be a path string or a list of ConllEntity objects")

    for original_conll_entity in source_entities:
        if len(augmented_sentences) >= max_new_sentences:
            print(f"Reached max_new_sentences limit: {max_new_sentences}")
            break

        original_tokens = original_conll_entity.tokens
        original_labels = original_conll_entity.labels

        current_sentence_pattern_types = tuple(
            get_entity_type_from_label(lbl) for lbl in original_labels)

        if current_sentence_pattern_types not in frequent_patterns:
            continue

        original_sentences_processed += 1

        for _ in range(augmentation_factor):
            if len(augmented_sentences) >= max_new_sentences:
                break

            new_tokens: list[str] = []
            new_labels: list[str] = []

            idx = 0
            made_a_replacement = False
            while idx < len(original_tokens):
                original_token = original_tokens[idx]
                original_label = original_labels[idx]
                current_entity_type = get_entity_type_from_label(
                    original_label)
                bio_prefix = original_label.split(
                    '-', 1)[0] if '-' in original_label else 'O'

                if bio_prefix in ['B', 'S'] and current_entity_type != 'O':
                    # This is the start of an entity
                    entity_tokens_original = [original_token]
                    entity_labels_original = [original_label]
                    temp_idx = idx + 1
                    if bio_prefix == 'B':
                        while temp_idx < len(original_tokens) and get_entity_type_from_label(original_labels[temp_idx]) == current_entity_type and original_labels[temp_idx].startswith(('I-', 'E-')):
                            entity_tokens_original.append(
                                original_tokens[temp_idx])
                            entity_labels_original.append(
                                original_labels[temp_idx])
                            if original_labels[temp_idx].startswith('E-'):
                                break
                            temp_idx += 1

                    # Try to replace this entity
                    if current_entity_type in available_entities and available_entities[current_entity_type]:
                        # Filter out the original entity itself to avoid replacing with the same
                        possible_replacements = [
                            e for e in available_entities[current_entity_type] if e != entity_tokens_original]
                        if possible_replacements:
                            replacement_entity_tokens = random.choice(
                                possible_replacements)
                            replacement_entity_labels = generate_bioes_labels(
                                replacement_entity_tokens, current_entity_type)

                            new_tokens.extend(replacement_entity_tokens)
                            new_labels.extend(replacement_entity_labels)
                            idx = temp_idx  # Move main index past the original entity
                            made_a_replacement = True
                            continue  # continue while loop
                        else:
                            # No other replacements available, use original
                            new_tokens.extend(entity_tokens_original)
                            new_labels.extend(entity_labels_original)
                            idx = temp_idx
                            continue
                    else:
                        # No entities of this type available for replacement, use original
                        new_tokens.extend(entity_tokens_original)
                        new_labels.extend(entity_labels_original)
                        idx = temp_idx
                        continue
                else:  # O tag or I/E tag (which are handled by B/S block)
                    new_tokens.append(original_token)
                    new_labels.append(original_label)
                    idx += 1

            # Ensure consistency
            if made_a_replacement and len(new_tokens) == len(new_labels):
                augmented_sentences.append(ConllEntity(
                    tokens=new_tokens, labels=new_labels))

    print(
        f"Processed {original_sentences_processed} original sentences matching frequent patterns.")
    return augmented_sentences


def write_conll_file(conll_entities: list[ConllEntity], output_file: str):
    """Writes a list of ConllEntity objects to a CoNLL formatted file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entity_obj in conll_entities:
            for token, label in zip(entity_obj.tokens, entity_obj.labels):
                f.write(f"{token} {label}\n")
            f.write("\n")  # Sentence separator
    print(f"Wrote {len(conll_entities)} sentences to {output_file}")


def extract_high_confidence_entities_from_predictions(
    # List of sentences, where each sentence is list of tokens
    sentences_tokens: list[list[str]],
    # List of predicted label sequences
    predicted_labels_sequences: list[list[str]],
    # List of logit tensors [seq_len, num_labels]
    logits_tensors: list[torch.Tensor],
    label_map_instance: LabelMap,
    confidence_threshold: float = 0.9
) -> dict[str, list[list[str]]]:
    """
    Extracts entities from model predictions that meet a confidence threshold.
    Returns a dictionary like: {'prov': [['北', '京', '市']], ...}
    """
    high_confidence_entities = collections.defaultdict(list)
    processed_entity_texts_for_type = collections.defaultdict(
        set)  # To avoid duplicates for same type

    if not (len(sentences_tokens) == len(predicted_labels_sequences) == len(logits_tensors)):
        print("Warning: Mismatch in lengths of input lists for confidence extraction. Skipping.")
        return high_confidence_entities

    for sent_idx, tokens in enumerate(sentences_tokens):
        labels = predicted_labels_sequences[sent_idx]
        logits = logits_tensors[sent_idx]

        if len(tokens) != len(labels) or len(tokens) != logits.shape[0]:
            # print(f"Warning: Mismatch in lengths for sentence {sent_idx}. Tokens: {len(tokens)}, Labels: {len(labels)}, Logits: {logits.shape[0]}. Skipping sentence.")
            continue

        probabilities = torch.softmax(logits, dim=-1)

        current_entity_tokens: list[str] = []
        current_entity_type: str | None = None
        current_entity_probs: list[float] = []

        for token_idx, token_str in enumerate(tokens):
            label = labels[token_idx]
            entity_type = get_entity_type_from_label(label)
            bio_prefix = label.split('-', 1)[0] if '-' in label else 'O'

            token_prob = probabilities[token_idx, label_map_instance.label2id.get(
                label, -1)].item() if label_map_instance.label2id.get(label, -1) != -1 else 0.0

            if bio_prefix == 'B':
                if current_entity_tokens and current_entity_type:  # Save previous if any
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    # Corrected usage for the entity being closed out
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)

                current_entity_tokens = [token_str]
                current_entity_type = entity_type
                current_entity_probs = [token_prob]
            elif bio_prefix == 'S':
                if current_entity_tokens and current_entity_type:  # Save previous if any
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    # Corrected usage for the entity being closed out
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)

                # Process S tag as a new, complete entity
                single_token_entity_tuple = tuple([token_str])
                if token_prob >= confidence_threshold and single_token_entity_tuple not in processed_entity_texts_for_type[entity_type]:
                    high_confidence_entities[entity_type].append([token_str])
                    processed_entity_texts_for_type[entity_type].add(
                        single_token_entity_tuple)
                current_entity_tokens = []
                current_entity_type = None
                current_entity_probs = []
            elif bio_prefix in ['I', 'E'] and current_entity_type == entity_type:
                current_entity_tokens.append(token_str)
                current_entity_probs.append(token_prob)
                if bio_prefix == 'E':
                    if current_entity_tokens and current_entity_type:
                        avg_prob = sum(
                            current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                        # Corrected usage for the entity being closed out by E
                        current_entity_text_tuple = tuple(
                            current_entity_tokens)
                        if avg_prob >= confidence_threshold and current_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                            high_confidence_entities[current_entity_type].append(
                                list(current_entity_tokens))
                            processed_entity_texts_for_type[current_entity_type].add(
                                current_entity_text_tuple)
                    current_entity_tokens = []
                    current_entity_type = None
                    current_entity_probs = []
            else:  # O tag or end of an entity without E (implicit end)
                if current_entity_tokens and current_entity_type:
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    # Corrected usage for the entity being implicitly closed out
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)
                current_entity_tokens = []
                current_entity_type = None
                current_entity_probs = []

        # After iterating all tokens in a sentence, save any trailing entity
        if current_entity_tokens and current_entity_type:
            avg_prob = sum(current_entity_probs) / \
                len(current_entity_probs) if current_entity_probs else 0
            # Corrected usage for trailing entity
            trailing_entity_text_tuple = tuple(current_entity_tokens)
            if avg_prob >= confidence_threshold and trailing_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                high_confidence_entities[current_entity_type].append(
                    list(current_entity_tokens))
                processed_entity_texts_for_type[current_entity_type].add(
                    trailing_entity_text_tuple)

    return high_confidence_entities


if __name__ == "__main__":
    # Load configuration
    app_config = AugmentConfig("augment.yaml")

    train_file = app_config.train_file
    dev_file = app_config.dev_file  # Assuming dev_file is in config
    label_map_instance = app_config.label_map  # Get LabelMap from AppConfig

    # --- First phase: Augment using ground truth from train_file ---
    print("--- Phase 1: Augmentation based on ground truth entities ---")
    print(f"Extracting entity type patterns from {train_file}...")
    frequent_patterns_gt = extract_entity_type_patterns(
        train_file, min_freq=app_config.augmentation_min_pattern_freq)
    print(
        f"Found {len(frequent_patterns_gt)} frequent patterns from ground truth.")

    print(f"Extracting entities by type from {train_file}...")
    entities_by_type_gt = extract_entities_by_type(train_file)
    print(f"Found ground truth entities for {len(entities_by_type_gt)} types.")

    print("Starting data augmentation by replacement (using ground truth entities)...")
    augmented_data_gt = augment_data_by_replacement(
        original_conll_file=train_file,
        frequent_patterns=frequent_patterns_gt,
        entities_by_type=entities_by_type_gt,
        label_map_instance=label_map_instance,
        augmentation_factor=app_config.augmentation_factor,
        max_new_sentences=app_config.augmentation_max_sentences
    )
    print(
        f"Generated {len(augmented_data_gt)} new sentences using ground truth entities.")

    augmented_output_file_gt = app_config.work_dir + "/train_augmented_gt.conll"
    if augmented_data_gt:
        write_conll_file(augmented_data_gt, augmented_output_file_gt)
    else:
        print("No data generated from ground truth augmentation.")

    # --- Second phase: Augment using high-confidence entities from model predictions ---
    print("\n--- Phase 2: Augmentation based on high-confidence predicted entities ---")

    # Initialize Predictor
    # The model_init_config for predictor should point to the base BERT model, not a trained checkpoint initially.
    # The actual trained model path is given to get_predictions_for_fold.
    # We need a config object that has .device, .label_map, and .model_name (for tokenizer in AddressNER)
    # Let's reuse app_config for simplicity, assuming its model_name is the base for tokenizer.
    # predictor_config = AppConfig(config_data) # Or a specific one if needed
    predictor = Predictor(model_init_config=app_config)

    # Define path to the trained model (e.g., from a specific fold or a single run)
    # This path needs to be correctly set based on your training output structure.
    # For example, if a single train run saves to config.work_dir/best_model.pt
    # Adjust if your models are in subdirs like fold_0, etc.
    trained_model_fold_dir = os.path.join(
        app_config.work_dir, app_config.model_name)
    # If you have k-folds, you might loop here or pick one fold's model.
    # For this example, assuming a single model path structure like `work_dir/best_model.pt` or `work_dir/swa_model.pt`

    print(
        f"Loading dev data ({dev_file}) for prediction to get high-confidence entities...")
    # The Predictor.get_predictions_for_fold expects a raw text file path if its internal loader is used.
    # If dev_file is CoNLL, we need to either convert it or ensure NERDataset handles it.
    # NERDataset in predictor.py is initialized with conll_examples. Let's assume dev_file can be used.
    # We also need the original tokens from dev_file to pass to extract_high_confidence_entities.

    dev_conll_reader = ConllReader()
    dev_sentences_conll = list(dev_conll_reader.read(dev_file))
    dev_sentences_tokens = [entity.tokens for entity in dev_sentences_conll]

    print(
        f"Predicting on {dev_file} using model from {trained_model_fold_dir}...")
    # Assuming use_swa_if_available is true if swa is generally used.
    use_swa = app_config.use_swa

    predicted_labels_dev, logits_dev = predictor.get_predictions_for_fold(
        fold_work_dir=trained_model_fold_dir,
        test_conll_examples=dev_sentences_conll,
        label_map=label_map_instance,
        batch_size=app_config.batch_size,
        use_swa_if_available=use_swa
    )

    if not predicted_labels_dev:
        print(
            "No predictions obtained from dev set. Skipping confidence-based augmentation.")
    else:
        print(f"Extracting high-confidence entities from dev set predictions...")
        # Ensure sentences_tokens align with predictions. `dev_sentences_tokens` from ConllReader should align if predictor processes dev.conll sequentially.
        # Check length alignment before proceeding
        if len(dev_sentences_tokens) != len(predicted_labels_dev):
            print(
                f"Mismatch between CoNLL read dev sentences ({len(dev_sentences_tokens)}) and predicted sentences ({len(predicted_labels_dev)}). CANNOT PROCEED WITH CONFIDENCE EXTRACTION.")
        else:
            # e.g., 0.9, add to config
            confidence_threshold = app_config.augmentation_confidence_threshold
            entities_by_type_conf = extract_high_confidence_entities_from_predictions(
                sentences_tokens=dev_sentences_tokens,
                predicted_labels_sequences=predicted_labels_dev,
                logits_tensors=logits_dev,
                label_map_instance=label_map_instance,
                confidence_threshold=confidence_threshold
            )
            print(
                f"Found {sum(len(v) for v in entities_by_type_conf.values())} high-confidence entities from dev set predictions.")
            for ent_type, ent_list in entities_by_type_conf.items():
                if ent_list:
                    print(
                        f"  Type: {ent_type}, Num high-conf entities: {len(ent_list)}, Example: {''.join(random.choice(ent_list))}")

            # Augment using the new high-confidence lexicon
            # We can choose to augment the original train_file or the already augmented train_augmented_gt.conll
            # Let's use original train_file for this example, and its patterns.
            if sum(len(v) for v in entities_by_type_conf.values()) > 0:
                print(
                    "Starting data augmentation by replacement (using high-confidence entities)...")
                augmented_data_conf = augment_data_by_replacement(
                    original_conll_file=train_file,  # Or use augmented_data_gt as a base
                    # Or re-calculate patterns if augmenting a different base
                    frequent_patterns=frequent_patterns_gt,
                    entities_by_type=entities_by_type_conf,
                    label_map_instance=label_map_instance,
                    augmentation_factor=app_config.augmentation_factor,  # Could be a different factor
                    max_new_sentences=app_config.augmentation_max_sentences  # Could be a different max
                )
                print(
                    f"Generated {len(augmented_data_conf)} new sentences using high-confidence entities.")

                augmented_output_file_conf = app_config.work_dir + "/train_augmented_conf.conll"
                if augmented_data_conf:
                    write_conll_file(augmented_data_conf,
                                     augmented_output_file_conf)
                else:
                    print("No data generated from confidence-based augmentation.")
            else:
                print(
                    "No high-confidence entities found, skipping second round of augmentation.")

    print("Augmentation script finished.")
