"""Tokenization class for F5-TTS."""

import os

from transformers.tokenization_utils import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def _is_chinese(char: str) -> bool:
    return "㄀" <= char <= "鿿"


class F5TTSTokenizer(PreTrainedTokenizer):
    r"""
    Constructs an F5-TTS tokenizer. F5-TTS is trained on a fixed, dataset-derived character vocabulary: Latin
    text is split character by character, and Chinese characters are converted to tone-marked pinyin syllables
    (falling back to the raw character when `pypinyin` is not installed) before being split. `vocab.txt` maps
    each resulting token to an id, with `" "` reserved for id `0`.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file, one token per line.
        pad_token (`str`, *optional*, defaults to `" "`):
            The token used for padding. F5-TTS pads text with `-1` rather than a vocabulary id; this is exposed
            only for [`PreTrainedTokenizer`] API compatibility.
        unk_token (`str`, *optional*, defaults to `" "`):
            The token substituted for characters absent from `vocab_file`. F5-TTS maps unknown characters to
            id `0`, the same id as `" "`.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids"]

    def __init__(self, vocab_file, pad_token=" ", unk_token=" ", **kwargs):
        with open(vocab_file, encoding="utf-8") as vocab_reader:
            self.vocab = {line.rstrip("\n"): index for index, line in enumerate(vocab_reader)}
        self.ids_to_tokens = {index: token for token, index in self.vocab.items()}
        super().__init__(pad_token=pad_token, unk_token=unk_token, **kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for char in text:
            if _is_chinese(char):
                tokens.append(" ")
                tokens.extend(_char_to_pinyin(char))
            else:
                tokens.append(char)
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token, 0))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        prefix = (filename_prefix + "-") if filename_prefix else ""
        vocab_file = os.path.join(save_directory, prefix + VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, _ in sorted(self.vocab.items(), key=lambda item: item[1]):
                writer.write(token + "\n")
        return (vocab_file,)


def _char_to_pinyin(char: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin

        return lazy_pinyin(char, style=Style.TONE3, tone_sandhi=True)
    except ImportError:
        return [char]


__all__ = ["F5TTSTokenizer"]
