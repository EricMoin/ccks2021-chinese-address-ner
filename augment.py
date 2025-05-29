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


def validate_bioes_sequence(labels: list[str]) -> tuple[bool, str]:
    """
    验证BIOES标签序列是否符合规范
    返回: (是否有效, 错误信息)
    """
    if not labels:
        return True, ""

    prev_bio = None
    prev_type = None

    for i, label in enumerate(labels):
        if label == 'O':
            prev_bio = 'O'
            prev_type = None
            continue

        if '-' not in label:
            return False, f"位置 {i}: 标签格式错误 '{label}'"

        bio_prefix, entity_type = label.split('-', 1)

        if bio_prefix == 'B':
            prev_bio = 'B'
            prev_type = entity_type
        elif bio_prefix == 'I':
            if prev_bio not in ['B', 'I'] or prev_type != entity_type:
                return False, f"位置 {i}: 孤立的I标签 '{label}', 前一个: '{labels[i-1] if i > 0 else 'None'}'"
            prev_bio = 'I'
            prev_type = entity_type
        elif bio_prefix == 'E':
            if prev_bio not in ['B', 'I'] or prev_type != entity_type:
                return False, f"位置 {i}: 孤立的E标签 '{label}', 前一个: '{labels[i-1] if i > 0 else 'None'}'"
            prev_bio = 'E'
            prev_type = None
        elif bio_prefix == 'S':
            prev_bio = 'S'
            prev_type = None
        else:
            return False, f"位置 {i}: 未知的BIO前缀 '{bio_prefix}'"

    return True, ""


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
            made_a_replacement = False

            # 重新实现：先正确解析所有实体，然后逐个处理
            # [(start_idx, end_idx, tokens, labels, entity_type)]
            entities = []

            # 第一步：正确识别所有实体边界
            i = 0
            while i < len(original_tokens):
                token = original_tokens[i]
                label = original_labels[i]
                entity_type = get_entity_type_from_label(label)
                bio_prefix = label.split('-', 1)[0] if '-' in label else 'O'

                if bio_prefix == 'B':
                    # 开始一个新实体
                    start_idx = i
                    entity_tokens = [token]
                    entity_labels = [label]
                    i += 1

                    # 继续收集这个实体的其余部分
                    while i < len(original_tokens):
                        next_token = original_tokens[i]
                        next_label = original_labels[i]
                        next_entity_type = get_entity_type_from_label(
                            next_label)
                        next_bio_prefix = next_label.split(
                            '-', 1)[0] if '-' in next_label else 'O'

                        # 如果是同一实体的I或E标签
                        if next_entity_type == entity_type and next_bio_prefix in ['I', 'E']:
                            entity_tokens.append(next_token)
                            entity_labels.append(next_label)
                            # 如果是E标签，实体结束
                            if next_bio_prefix == 'E':
                                end_idx = i  # E标签的位置就是结束位置
                                i += 1  # 移动到下一个位置
                                break
                            i += 1
                        else:
                            # 不是同一实体，当前实体结束（没有显式E标签）
                            end_idx = i - 1  # 前一个位置是实体的结束位置
                            break
                    else:
                        # while循环正常结束，实体在句子末尾结束
                        end_idx = i - 1

                    entities.append(
                        (start_idx, end_idx, entity_tokens, entity_labels, entity_type))

                elif bio_prefix == 'S':
                    # 单字符实体
                    entities.append((i, i, [token], [label], entity_type))
                    i += 1
                else:
                    # O标签或其他，跳过
                    i += 1

            # 第二步：处理每个token，如果是实体则尝试替换
            processed_until = 0

            for start_idx, end_idx, entity_tokens, entity_labels, entity_type in entities:
                # 添加实体之前的非实体token
                while processed_until < start_idx:
                    new_tokens.append(original_tokens[processed_until])
                    new_labels.append(original_labels[processed_until])
                    processed_until += 1

                # 尝试替换当前实体
                if entity_type in available_entities and available_entities[entity_type]:
                    possible_replacements = [
                        e for e in available_entities[entity_type] if e != entity_tokens]
                    if possible_replacements:
                        replacement_entity_tokens = random.choice(
                            possible_replacements)
                        replacement_entity_labels = generate_bioes_labels(
                            replacement_entity_tokens, entity_type)

                        new_tokens.extend(replacement_entity_tokens)
                        new_labels.extend(replacement_entity_labels)
                        made_a_replacement = True
                    else:
                        # 没有可替换的实体，保持原样
                        new_tokens.extend(entity_tokens)
                        # MODIFICATION START: Regenerate labels for consistency even if not replaced
                        # Original line: new_labels.extend(entity_labels)
                        refreshed_entity_labels = generate_bioes_labels(
                            entity_tokens, entity_type)
                        new_labels.extend(refreshed_entity_labels)
                        # If refreshing labels changed them, it's a modification.
                        # This ensures that if original entity_labels were subtly incorrect (e.g. E for single token)
                        # they get corrected to proper S-tag etc.
                        if tuple(refreshed_entity_labels) != tuple(entity_labels):
                            made_a_replacement = True  # Treat as a replacement for validation logic
                        # MODIFICATION END
                else:
                    # 没有该类型的实体，保持原样
                    new_tokens.extend(entity_tokens)
                    # MODIFICATION START: Regenerate labels for consistency here as well
                    # Original line: new_labels.extend(entity_labels)
                    refreshed_entity_labels = generate_bioes_labels(
                        entity_tokens, entity_type)
                    new_labels.extend(refreshed_entity_labels)
                    if tuple(refreshed_entity_labels) != tuple(entity_labels):
                        made_a_replacement = True
                    # MODIFICATION END

                processed_until = end_idx + 1

            # 添加剩余的非实体token
            while processed_until < len(original_tokens):
                new_tokens.append(original_tokens[processed_until])
                new_labels.append(original_labels[processed_until])
                processed_until += 1

            # Ensure consistency and validate BIOES sequence
            if made_a_replacement and len(new_tokens) == len(new_labels):
                # 验证生成的标签序列
                is_valid, error_msg = validate_bioes_sequence(new_labels)
                if is_valid:
                    augmented_sentences.append(ConllEntity(
                        tokens=new_tokens, labels=new_labels))
                else:
                    print(f"Warning: Invalid BIOES sequence detected and skipped.")
                    print(f"  Tokens: {''.join(new_tokens)}")
                    print(f"  Labels: {new_labels}")
                    print(f"  Error: {error_msg}")
                    print(
                        f"  Original: {''.join(original_tokens)} -> {original_labels}")

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

    # 添加调试信息
    print(f"Debug: Processing {len(sentences_tokens)} sentences")
    print(f"Debug: Confidence threshold: {confidence_threshold}")
    print(f"Debug: Label map has {len(label_map_instance.label2id)} labels")

    # 统计预测标签分布
    all_predicted_labels = []
    for labels in predicted_labels_sequences:
        all_predicted_labels.extend(labels)

    label_counts = collections.Counter(all_predicted_labels)
    print(f"Debug: Top 10 predicted labels: {label_counts.most_common(10)}")

    # 检查非O标签的数量
    non_o_labels = [label for label in all_predicted_labels if label != 'O']
    print(
        f"Debug: Total non-O labels: {len(non_o_labels)} out of {len(all_predicted_labels)}")

    if len(non_o_labels) == 0:
        print("Debug: No non-O labels found in predictions! All predictions are 'O'.")
        return high_confidence_entities

    # 检查标签映射
    sample_labels = list(set(all_predicted_labels))[:10]
    print(f"Debug: Sample label mappings:")
    for label in sample_labels:
        label_id = label_map_instance.label2id.get(label, -1)
        print(f"  {label} -> {label_id}")

    entities_found = 0
    high_conf_entities_found = 0

    # 添加长度统计调试信息
    length_mismatches = 0
    processed_sentences = 0

    for sent_idx, tokens in enumerate(sentences_tokens):
        labels = predicted_labels_sequences[sent_idx]
        logits = logits_tensors[sent_idx]

        if len(tokens) != len(labels):
            print(
                f"Warning: Mismatch in lengths for sentence {sent_idx}. Tokens: {len(tokens)}, Labels: {len(labels)}. Skipping sentence.")
            continue

        # 修复：处理logits长度与实际序列长度不匹配的问题
        actual_seq_len = len(tokens)
        if logits.shape[0] < actual_seq_len:
            print(
                f"Warning: Logits too short for sentence {sent_idx}. Logits: {logits.shape[0]}, Tokens: {actual_seq_len}. Skipping sentence.")
            length_mismatches += 1
            continue

        # 只使用实际序列长度的logits
        logits_actual = logits[:actual_seq_len, :]
        probabilities = torch.softmax(logits_actual, dim=-1)
        processed_sentences += 1

        # 调试：打印前几个句子的长度信息
        if sent_idx < 5:
            print(
                f"Debug: Sentence {sent_idx} - Tokens: {len(tokens)}, Labels: {len(labels)}, Logits: {logits.shape[0]} -> {actual_seq_len}")

        current_entity_tokens: list[str] = []
        current_entity_type: str | None = None
        current_entity_probs: list[float] = []

        for token_idx, token_str in enumerate(tokens):
            label = labels[token_idx]
            entity_type = get_entity_type_from_label(label)
            bio_prefix = label.split('-', 1)[0] if '-' in label else 'O'

            # 获取标签ID和置信度
            label_id = label_map_instance.label2id.get(label, -1)
            if label_id == -1:
                token_prob = 0.0
                if sent_idx < 3:  # 只在前几个句子中打印调试信息
                    print(
                        f"Debug: Unknown label '{label}' at sentence {sent_idx}, token {token_idx}")
            else:
                token_prob = probabilities[token_idx, label_id].item()

            # 调试：打印前几个非O标签的置信度
            if bio_prefix != 'O' and sent_idx < 3:
                print(
                    f"Debug: Sentence {sent_idx}, Token '{token_str}', Label '{label}', Confidence: {token_prob:.4f}")

            if bio_prefix == 'B':
                # 保存之前的实体（如果有）
                if current_entity_tokens and current_entity_type:
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    entities_found += 1
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)
                        high_conf_entities_found += 1
                        print(
                            f"Debug: Found high-conf entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")

                # 开始新实体
                current_entity_tokens = [token_str]
                current_entity_type = entity_type
                current_entity_probs = [token_prob]
            elif bio_prefix == 'S':
                # 保存之前的实体（如果有）
                if current_entity_tokens and current_entity_type:
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    entities_found += 1
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)
                        high_conf_entities_found += 1
                        print(
                            f"Debug: Found high-conf entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")

                # 处理单字符实体
                entities_found += 1
                single_token_entity_tuple = tuple([token_str])
                if token_prob >= confidence_threshold and single_token_entity_tuple not in processed_entity_texts_for_type[entity_type]:
                    high_confidence_entities[entity_type].append([token_str])
                    processed_entity_texts_for_type[entity_type].add(
                        single_token_entity_tuple)
                    high_conf_entities_found += 1
                    print(
                        f"Debug: Found high-conf S entity: {token_str} ({entity_type}) with confidence {token_prob:.4f}")
                current_entity_tokens = []
                current_entity_type = None
                current_entity_probs = []
            elif bio_prefix in ['I', 'E']:
                # 修复：更保守地处理I-和E-标签
                if current_entity_type == entity_type:
                    # 继续当前实体（正常情况）
                    current_entity_tokens.append(token_str)
                    current_entity_probs.append(token_prob)
                    if bio_prefix == 'E':
                        # 结束当前实体
                        if current_entity_tokens and current_entity_type:
                            avg_prob = sum(
                                current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                            entities_found += 1
                            current_entity_text_tuple = tuple(
                                current_entity_tokens)
                            if avg_prob >= confidence_threshold and current_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                                high_confidence_entities[current_entity_type].append(
                                    list(current_entity_tokens))
                                processed_entity_texts_for_type[current_entity_type].add(
                                    current_entity_text_tuple)
                                high_conf_entities_found += 1
                                print(
                                    f"Debug: Found high-conf E entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")
                        current_entity_tokens = []
                        current_entity_type = None
                        current_entity_probs = []
                elif current_entity_type is None and bio_prefix == 'I':
                    # 孤立的I-标签：更保守的处理策略
                    # 检查是否在句子开头（可能是跨句子的实体片段）
                    if token_idx == 0:
                        # 句子开头的孤立I-标签，很可能是实体被分割，跳过
                        print(
                            f"Debug: Skipping orphaned I- tag at sentence start {sent_idx}, token {token_idx}: {label}")
                        continue
                    else:
                        # 句子中间的孤立I-标签，可能是标注错误，也跳过
                        print(
                            f"Debug: Skipping orphaned I- tag in middle {sent_idx}, token {token_idx}: {label}")
                        continue
                else:
                    # 实体类型不匹配或其他情况
                    if current_entity_tokens and current_entity_type:
                        # 结束之前的实体
                        avg_prob = sum(
                            current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                        entities_found += 1
                        previous_entity_text_tuple = tuple(
                            current_entity_tokens)
                        if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                            high_confidence_entities[current_entity_type].append(
                                list(current_entity_tokens))
                            processed_entity_texts_for_type[current_entity_type].add(
                                previous_entity_text_tuple)
                            high_conf_entities_found += 1
                            print(
                                f"Debug: Found high-conf implicit entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")

                    # 对于孤立的E-标签，检查是否应该作为单字符实体
                    if bio_prefix == 'E':
                        # 只有当它不是句子开头且置信度足够高时，才考虑作为单字符实体
                        if token_idx > 0 and token_prob >= confidence_threshold:
                            entities_found += 1
                            single_token_entity_tuple = tuple([token_str])
                            if single_token_entity_tuple not in processed_entity_texts_for_type[entity_type]:
                                high_confidence_entities[entity_type].append(
                                    [token_str])
                                processed_entity_texts_for_type[entity_type].add(
                                    single_token_entity_tuple)
                                high_conf_entities_found += 1
                                print(
                                    f"Debug: Found high-conf orphaned E entity: {token_str} ({entity_type}) with confidence {token_prob:.4f}")
                        else:
                            print(
                                f"Debug: Skipping suspicious orphaned E- tag at sentence {sent_idx}, token {token_idx}: {label}")

                    # 重置状态，不开始新实体
                    current_entity_tokens = []
                    current_entity_type = None
                    current_entity_probs = []
            else:  # O tag
                if current_entity_tokens and current_entity_type:
                    avg_prob = sum(
                        current_entity_probs) / len(current_entity_probs) if current_entity_probs else 0
                    entities_found += 1
                    previous_entity_text_tuple = tuple(current_entity_tokens)
                    if avg_prob >= confidence_threshold and previous_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                        high_confidence_entities[current_entity_type].append(
                            list(current_entity_tokens))
                        processed_entity_texts_for_type[current_entity_type].add(
                            previous_entity_text_tuple)
                        high_conf_entities_found += 1
                        print(
                            f"Debug: Found high-conf implicit entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")
                current_entity_tokens = []
                current_entity_type = None
                current_entity_probs = []

        # After iterating all tokens in a sentence, save any trailing entity
        if current_entity_tokens and current_entity_type:
            avg_prob = sum(current_entity_probs) / \
                len(current_entity_probs) if current_entity_probs else 0
            entities_found += 1
            trailing_entity_text_tuple = tuple(current_entity_tokens)
            if avg_prob >= confidence_threshold and trailing_entity_text_tuple not in processed_entity_texts_for_type[current_entity_type]:
                high_confidence_entities[current_entity_type].append(
                    list(current_entity_tokens))
                processed_entity_texts_for_type[current_entity_type].add(
                    trailing_entity_text_tuple)
                high_conf_entities_found += 1
                print(
                    f"Debug: Found high-conf trailing entity: {''.join(current_entity_tokens)} ({current_entity_type}) with confidence {avg_prob:.4f}")

    print(f"Debug: Total entities found: {entities_found}")
    print(f"Debug: High-confidence entities found: {high_conf_entities_found}")
    print(
        f"Debug: Processed sentences: {processed_sentences}/{len(sentences_tokens)}")
    print(f"Debug: Length mismatches: {length_mismatches}")

    return high_confidence_entities


if __name__ == "__main__":
    # Load configuration
    app_config = AugmentConfig("augment.yaml")

    train_file = app_config.train_file
    dev_file = app_config.dev_file
    label_map_instance = app_config.label_map

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

    # 数据源选择策略：
    # 1. 优先使用dev集（安全，有ground truth验证）
    # 2. 可选使用测试集（更多数据，但需要谨慎）
    # 3. 理想情况：使用外部无标签数据

    # 从配置文件读取伪标签策略
    use_test_for_pseudo_labeling = app_config.use_test_for_pseudo_labeling
    pseudo_label_source = "test_file" if use_test_for_pseudo_labeling else "dev_file"

    if use_test_for_pseudo_labeling:
        print("⚠️  WARNING: Using test set for pseudo-labeling!")
        print("   This may lead to data leakage and overly optimistic evaluation.")
        print("   Consider using external unlabeled data instead.")
        pseudo_file = app_config.test_file
        print(f"   Using test file: {pseudo_file}")
    else:
        print("✅ Using dev set for pseudo-labeling (recommended for safety)")
        pseudo_file = dev_file
        print(f"   Using dev file: {pseudo_file}")

    # Initialize Predictor
    predictor = Predictor(model_init_config=app_config)

    # Define path to the trained model
    trained_model_fold_dir = os.path.join(
        app_config.work_dir, app_config.model_name)

    print(
        f"Loading data from {pseudo_file} for prediction to get high-confidence entities...")

    # 根据数据源类型选择不同的处理方式
    if use_test_for_pseudo_labeling:
        # 对于测试集，需要特殊处理（假设是原始文本格式）
        print("Processing test file format...")
        # 这里需要根据实际的测试文件格式进行调整
        # 假设测试文件格式为: guid\u0001text
        test_sentences_char_tokens = []
        if os.path.exists(pseudo_file):
            with open(pseudo_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            text_part = line.split('\u0001')[1]
                            test_sentences_char_tokens.append(list(text_part))
                        except IndexError:
                            print(f"Warning: Skipping malformed line: {line}")
                            test_sentences_char_tokens.append([])

        # 创建伪ConLL实体（没有真实标签）
        pseudo_conll_examples = [ConllEntity(chars, ['O'] * len(chars))
                                 for chars in test_sentences_char_tokens]
        pseudo_sentences_tokens = test_sentences_char_tokens
    else:
        # 对于dev集，使用ConLL格式
        print("Processing CoNLL format...")
        dev_conll_reader = ConllReader()
        pseudo_conll_examples = list(dev_conll_reader.read(pseudo_file))
        pseudo_sentences_tokens = [
            entity.tokens for entity in pseudo_conll_examples]

    print(
        f"Predicting on {pseudo_file} using model from {trained_model_fold_dir}...")
    use_swa = app_config.use_swa

    predicted_labels_pseudo, logits_pseudo = predictor.get_predictions_for_conll_data(
        fold_work_dir=trained_model_fold_dir,
        test_conll_examples=pseudo_conll_examples,
        label_map=label_map_instance,
        batch_size=app_config.batch_size,
        use_swa_if_available=use_swa
    )

    if not predicted_labels_pseudo:
        print("No predictions obtained from pseudo-labeling source. Skipping confidence-based augmentation.")
    else:
        print(f"Extracting high-confidence entities from predictions...")

        # 验证数据对齐
        if len(pseudo_sentences_tokens) != len(predicted_labels_pseudo):
            print(
                f"Mismatch between source sentences ({len(pseudo_sentences_tokens)}) and predicted sentences ({len(predicted_labels_pseudo)}). CANNOT PROCEED WITH CONFIDENCE EXTRACTION.")
        else:
            confidence_threshold = app_config.augmentation_confidence_threshold
            entities_by_type_conf = extract_high_confidence_entities_from_predictions(
                sentences_tokens=pseudo_sentences_tokens,
                predicted_labels_sequences=predicted_labels_pseudo,
                logits_tensors=logits_pseudo,
                label_map_instance=label_map_instance,
                confidence_threshold=confidence_threshold
            )

            print(
                f"Found {sum(len(v) for v in entities_by_type_conf.values())} high-confidence entities from {pseudo_label_source} predictions.")

            # 如果使用dev集，可以进行质量验证
            if not use_test_for_pseudo_labeling and hasattr(app_config, 'validate_pseudo_labels') and app_config.validate_pseudo_labels:
                print("Validating pseudo-labels against ground truth...")
                # 这里可以添加验证逻辑，比较预测标签与真实标签的一致性

            for ent_type, ent_list in entities_by_type_conf.items():
                if ent_list:
                    print(
                        f"  Type: {ent_type}, Num high-conf entities: {len(ent_list)}, Example: {''.join(random.choice(ent_list))}")

            # 使用高置信度实体进行数据增强
            if sum(len(v) for v in entities_by_type_conf.values()) > 0:
                print(
                    "Starting data augmentation by replacement (using high-confidence entities)...")
                augmented_data_conf = augment_data_by_replacement(
                    original_conll_file=train_file,
                    frequent_patterns=frequent_patterns_gt,
                    entities_by_type=entities_by_type_conf,
                    label_map_instance=label_map_instance,
                    augmentation_factor=app_config.augmentation_factor,
                    max_new_sentences=app_config.augmentation_max_sentences
                )
                print(
                    f"Generated {len(augmented_data_conf)} new sentences using high-confidence entities.")

                suffix = "test" if use_test_for_pseudo_labeling else "dev"
                augmented_output_file_conf = f"{app_config.work_dir}/train_augmented_conf_{suffix}.conll"
                if augmented_data_conf:
                    write_conll_file(augmented_data_conf,
                                     augmented_output_file_conf)
                else:
                    print("No data generated from confidence-based augmentation.")
            else:
                print(
                    "No high-confidence entities found, skipping second round of augmentation.")

    print("Augmentation script finished.")
