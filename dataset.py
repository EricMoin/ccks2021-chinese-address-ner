import torch
from torch.utils.data import Dataset
from conll_reader import ConllEntity
from transformers import AutoTokenizer


class NERDataEntity:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def to_map(self):
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels
        }


class NERDataset(Dataset):
    def __init__(self, data: list[ConllEntity], tokenizer: AutoTokenizer, label_map: dict[str, int]):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __getitem__(self, index: int) -> dict:
        # Initialize empty lists
        tokens = []
        labels = []
        is_prompt = False

        # Process all tokens/labels
        for token, label in zip(self.data[index].tokens, self.data[index].labels):
            if token == "<EOS>":
                is_prompt = True
                continue
            if not is_prompt:
                tokens.append(token)
                labels.append(label)

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=150,
            is_split_into_words=True,
            return_tensors="pt",
        )

        # Convert labels to tensor with fallback to "O" for unknown labels
        label_ids = []
        for label in labels:
            if label in self.label_map:
                label_ids.append(self.label_map[label])
            else:
                # Use "O" (Outside) tag for unknown labels
                label_ids.append(self.label_map["O"])
                print(
                    f"Warning: Unknown label '{label}' found, using 'O' instead")

        # Pad label_ids to match input_ids length
        padded_labels = label_ids + \
            [self.label_map["O"]] * (150 - len(label_ids))

        return NERDataEntity(
            input_ids=encoding["input_ids"].squeeze(0),
            attention_mask=encoding["attention_mask"].squeeze(0),
            labels=torch.tensor(padded_labels)
        ).to_map()

    def __len__(self):
        return len(self.data)


class NERTestDataset(Dataset):
    def __init__(self, test_file: str, tokenizer: AutoTokenizer, label_map: dict[str, int]):
        super().__init__()
        self.tokenizer = tokenizer
        self.examples = []

        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Remove line number prefix and strip whitespace
                # Format: "1朝阳区小关北里000-0号" -> "朝阳区小关北里000-0号"
                text = line.strip()
                if text:
                    # Remove the line number at the beginning
                    text_without_number = ''.join(c for i, c in enumerate(
                        text) if not (i == 0 and c.isdigit()))
                self.examples.append(text_without_number)

    def __getitem__(self, index: int) -> dict:
        text = self.examples[index]

        # Convert text to character-level tokens for Chinese
        tokens = list(text)

        # Tokenize the characters
        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=150,
            is_split_into_words=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "text": text,  # Include original text for reference
            "tokens": tokens  # Include original tokens for mapping predictions back
        }

    def __len__(self):
        return len(self.examples)
