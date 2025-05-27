import logging
import os

logger = logging.getLogger(__name__)


class SentenceReader:

    def read_tokens(self, file_path: str) -> list[list[str]]:
        test_sentences_tokens = []

        with open(file_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if line:
                    parts = line.split('\u0001', 1)
                    test_sentences_tokens.append(list(parts[1]))
        logger.info(
            f"Extracted {len(test_sentences_tokens)} token sequences from {file_path}")
        return test_sentences_tokens

    def read_parts(self, file_path: str) -> list[tuple[str, str]]:
        sentence_parts = []
        with open(file_path, 'r', encoding='utf-8') as f_orig:
            for line_idx, line in enumerate(f_orig):
                line = line.strip()
                if line:
                    parts = line.split('\u0001', 1)
                    if len(parts) == 2:
                        sentence_parts.append((parts[0], parts[1]))
                    else:
                        logger.warning(
                            f"Malformed line {line_idx+1} in original test file {file_path}: '{line}'. Skipping.")
        return sentence_parts

    def read_corpus(self, file_path: str) -> list[str]:
        texts = []
        if not os.path.exists(file_path):
            logger.error(f"Corpus file not found: {file_path}")
            return texts
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    texts.append(line)
        return texts
