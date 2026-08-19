"""LibriTTS-P dataset implementation.

LibriTTS-P extends LibriTTS with style and speaker prompts for controllable TTS.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from torch import Tensor
from torchaudio._internal import download_url_to_file
from torchaudio.datasets import LIBRITTS
from torchaudio.datasets.libritts import load_libritts_item, _CHECKSUMS


_METADATA_FILENAME = "metadata_w_style_prompt_tags_v230922.csv"
_STYLE_FILENAME = "style_prompt_candidates_v230922.csv"

_BASE_URL = "https://raw.githubusercontent.com/line/LibriTTS-P/main/data/"
_URLS = {
    "metadata": _BASE_URL + "metadata_w_style_prompt_tags_v230922.csv",
    "style": _BASE_URL + "style_prompt_candidates_v230922.csv",
    "df1": _BASE_URL + "df1_en.csv",
    "df2": _BASE_URL + "df2_en.csv",
    "df3": _BASE_URL + "df3_en.csv",
}


def download_libritts_audio(root: Union[str, Path], url: str) -> None:
    """Downloads and extracts the base LibriTTS audio archive.

    Args:
        root (str or Path): Root directory where the dataset will be stored.
        url (str): Dataset split (e.g., "dev-clean", "train-clean-100").
    """
    root = os.fspath(root)
    base_url = "http://www.openslr.org/resources/60/"
    ext_archive = ".tar.gz"
    full_url = os.path.join(base_url, url + ext_archive)
    
    archive_path = os.path.join(root, url + ext_archive)
    target_path = os.path.join(root, "LibriTTS", url)
    
    if not os.path.isdir(target_path):
        if not os.path.isfile(archive_path):
            checksum = _CHECKSUMS.get(full_url)
            print(f"[INFO] Downloading LibriTTS audio: {full_url}")
            download_url_to_file(full_url, archive_path, hash_prefix=checksum)
        
        print(f"[INFO] Extracting LibriTTS audio: {archive_path}")
        from torchaudio.datasets.utils import _extract_tar
        _extract_tar(archive_path)


def get_libritts_p_metadata_paths(root: Union[str, Path]) -> Dict[str, str]:
    """Resolves absolute paths for LibriTTS-P metadata files.

    Args:
        root (str or Path): Root directory of the dataset.

    Returns:
        Dict[str, str]: Dictionary containing paths for 'metadata' and 'style' CSVs.
    """
    root = os.fspath(root)
    paths = {
        "metadata": os.path.join(root, _METADATA_FILENAME),
        "style": os.path.join(root, _STYLE_FILENAME),
    }
    
    for key in ["metadata", "style"]:
        if not os.path.exists(paths[key]):
            alt_path = os.path.join(root, "LibriTTS", os.path.basename(paths[key]))
            if os.path.exists(alt_path):
                paths[key] = alt_path
    return paths

def download_libritts_p_metadata(root: Union[str, Path], annotator: Optional[str] = None) -> None:
    """Downloads LibriTTS-P specific metadata and annotator CSVs.

    Args:
        root (str or Path): Root directory for storage.
        annotator (str, optional): Speaker prompt annotator (e.g., "df1").
    """
    root = os.fspath(root)
    paths = get_libritts_p_metadata_paths(root)
    
    os.makedirs(root, exist_ok=True)
    
    if not os.path.isfile(paths["metadata"]):
        os.makedirs(os.path.dirname(paths["metadata"]), exist_ok=True)
        download_url_to_file(_URLS["metadata"], paths["metadata"])
    if not os.path.isfile(paths["style"]):
        os.makedirs(os.path.dirname(paths["style"]), exist_ok=True)
        download_url_to_file(_URLS["style"], paths["style"])

    if annotator is not None:
        speaker_path = os.path.join(root, f"{annotator}_en.csv")
        if not os.path.isfile(speaker_path):
            os.makedirs(os.path.dirname(speaker_path), exist_ok=True)
            download_url_to_file(_URLS[annotator], speaker_path)

def load_libritts_p_prompts(
    root: Union[str, Path], 
    annotator: Optional[str] = None
) -> Tuple[Dict[str, Tuple[List[str], int]], Optional[Dict[int, List[str]]]]:
    """Loads style and speaker prompts into memory-efficient lookups.

    Args:
        root (str or Path): Root directory of the dataset.
        annotator (str, optional): Speaker prompt annotator.

    Returns:
        Tuple:
            - Dict[str, Tuple[List[str], int]]: Mapping from utterance ID to 
              (style prompts, speaker ID).
            - Optional[Dict[int, List[str]]]: Mapping from speaker ID to 
              speaker trait variants.
    """
    paths = get_libritts_p_metadata_paths(root)
    
    # Load style prompts
    style_df = pd.read_csv(
        paths["style"],
        sep='|',
        header=None,
        names=['style_prompt_key', 'style_prompt'],
    )
    style_prompts_by_key = {
        row.style_prompt_key: row.style_prompt.split(";")
        for row in style_df.itertuples(index=False)
    }

    # Load utterance metadata
    metadata_df = pd.read_csv(paths["metadata"])
    prompts = {}

    for row in metadata_df.itertuples(index=False):
        utterance_id = row.item_name
        style_prompt_key = row.style_prompt_key
        speaker_id = int(row.spk_id)

        if style_prompt_key in style_prompts_by_key:
            style_prompts = style_prompts_by_key[style_prompt_key]
            prompts[utterance_id] = (style_prompts, speaker_id)

    # Load speaker prompts
    speaker_prompts = None
    if annotator is not None:
        speaker_path = os.path.join(os.fspath(root), f"{annotator}_en.csv")
        if not os.path.isfile(speaker_path):
            alt_path = os.path.join(os.fspath(root), "LibriTTS", f"{annotator}_en.csv")
            if os.path.exists(alt_path):
                speaker_path = alt_path
                
        if not os.path.isfile(speaker_path):
             raise RuntimeError(f"Speaker prompt file not found for {annotator}")

        speaker_df = pd.read_csv(
            speaker_path,
            sep='|',
            header=None,
            names=['spk_id', 'speaker_prompt'],
        )
        speaker_prompts = {
            int(row.spk_id): row.speaker_prompt.split(",")
            for row in speaker_df.itertuples(index=False)
        }
    
    return prompts, speaker_prompts


class LIBRITTS_P(LIBRITTS):
    """LibriTTS-P dataset with style and speaker prompts.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): Dataset split. Allowed values: ``"dev-clean"``, 
            ``"dev-other"``, ``"test-clean"``, ``"test-other"``, ``"train-clean-100"``, 
            ``"train-clean-360"``, ``"train-other-500"``.
            (default: ``"train-clean-100"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriTTS"``)
        download (bool, optional):
            Whether to download the dataset if it is not found. (default: ``False``).
        annotator (str or None, optional):
            Speaker prompt annotator. Must be one of ``["df1", "df2", "df3"]`` or ``None``.
            (default: ``None``)
    """

    def __init__(
        self,
        root: Union[str, Path],
        url: str = "train-clean-100",
        folder_in_archive: str = "LibriTTS",
        download: bool = False,
        annotator: Optional[str] = None,
    ) -> None:
        super().__init__(root, url, folder_in_archive, download)

        self._annotator = annotator
        
        if annotator is not None and annotator not in ["df1", "df2", "df3"]:
            raise ValueError(f"annotator must be one of ['df1', 'df2', 'df3'] or None")

        if download:
            download_libritts_p_metadata(root, annotator)

        paths = get_libritts_p_metadata_paths(root)
        self._metadata_path = paths["metadata"]
        self._style_path = paths["style"]

        if not os.path.isfile(self._metadata_path) or not os.path.isfile(self._style_path):
            raise RuntimeError("Metadata not found. Set download=True.")

        self._prompts, self._speaker_prompts = load_libritts_p_prompts(root, annotator)
        self._filter_valid_samples()


    def _filter_valid_samples(self) -> None:
        """Filter valid samples during initialization."""
        self._valid_indices = []

        for i in range(len(self._walker)):
            fileid = self._walker[i]

            if fileid not in self._prompts:
                continue

            _, metadata_speaker_id = self._prompts[fileid]

            try:
                parts = fileid.split("_")
                if len(parts) < 4:
                    continue
                file_speaker_id = int(parts[0])
            except (ValueError, IndexError):
                continue

            if metadata_speaker_id != file_speaker_id:
                continue

            if self._speaker_prompts is not None:
                if file_speaker_id not in self._speaker_prompts:
                    continue

            self._valid_indices.append(i)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, int, int, str, List[str], Optional[List[str]]]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform
            int:
                Sample rate
            str:
                Original text
            str:
                Normalized text
            int:
                Speaker ID
            int:
                Chapter ID
            str:
                Utterance ID
            List[str]:
                Style prompts (all variants)
            List[str] or None:
                Speaker prompts (all variants) if annotator was specified
        """
        actual_idx = self._valid_indices[n]
        fileid = self._walker[actual_idx]

        waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id = \
            load_libritts_item(
                fileid,
                self._path,
                self._ext_audio,
                self._ext_original_txt,
                self._ext_normalized_txt,
            )

        style_prompts, _ = self._prompts[utterance_id]

        speaker_prompts = None
        if self._speaker_prompts is not None:
            speaker_prompts = self._speaker_prompts[speaker_id]

        return waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id, \
            style_prompts, speaker_prompts
