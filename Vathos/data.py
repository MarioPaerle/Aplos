from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from pathlib import Path


class TextDataset(Dataset):
    """Text dataset supporting lists, files, and folder structures."""

    def __init__(self, source: Union[List[str], str, Path]):
        """
        Args:
            source: Can be:
                - List of strings (text data)
                - Path to single .txt file
                - Path to folder containing .txt files (recursive)
        """
        self.texts = []
        self.file_paths = []  # Track source files for debugging

        if isinstance(source, list):
            self.texts = source
            self.file_paths = [None] * len(source)
        else:
            source = Path(source)
            if source.is_file():
                self._load_file(source)
            elif source.is_dir():
                self._load_directory(source)
            else:
                raise ValueError(f"Source not found: {source}")

        if len(self.texts) == 0:
            raise ValueError("No text data loaded!")

    def _load_file(self, file_path: Path):
        """Load a single text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                self.texts.append(text)
                self.file_paths.append(str(file_path))

    def _load_directory(self, dir_path: Path):
        """Recursively load all .txt files from directory."""
        txt_files = sorted(dir_path.rglob("*.txt"))

        for file_path in txt_files:
            try:
                self._load_file(file_path)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

    def get_file_path(self, idx: int) -> Optional[str]:
        """Get source file path for debugging."""
        return self.file_paths[idx]


@dataclass
class TokenizerConfig:
    """Tokenizer configuration."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"  # or "longest"
    return_tensors: str = "pt"


class TokenizerWrapper:
    """Wrapper for HuggingFace tokenizer with collating dataloader."""

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

    def collate_fn(self, batch: List[str]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        return self.tokenizer(
            batch,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors=self.config.return_tensors
        )

    def decode(self, token_ids, skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.

        Args:
            token_ids: Single sequence or batch of token IDs (tensor or list)
            skip_special_tokens: Whether to remove [CLS], [SEP], [PAD] etc.

        Returns:
            Decoded text string or list of strings
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a batch of token IDs back to texts.

        Args:
            token_ids: Batch of token IDs (shape: [batch_size, seq_len])
            skip_special_tokens: Whether to remove special tokens

        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_dataloader(
        self,
        source: Union[List[str], str, Path],
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader with tokenization collation.

        Args:
            source: List of texts, file path, or directory path
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
        """
        dataset = TextDataset(source)
        print(f"Loaded {len(dataset)} texts")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )


if __name__ == "__main__":
    texts = "_tests"

    config = TokenizerConfig(
        model_name="bert-base-uncased",
        max_length=32,
        padding="longest"
    )
    wrapper = TokenizerWrapper(config)

    print("=== Example 1: List of texts ===")
    dataloader = wrapper.get_dataloader(texts, batch_size=2, shuffle=False)

    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}: {batch['input_ids'].shape}")

        decoded_batch = wrapper.batch_decode(batch['input_ids'], skip_special_tokens=True)
        print(f"Decoded texts: {decoded_batch}")

        single_decoded = wrapper.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"First sequence: {single_decoded}")

        with_special = wrapper.decode(batch['input_ids'][0], skip_special_tokens=False)
        print(f"With special tokens: {with_special}")