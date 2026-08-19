"""LibriTTS-P Curated Dataset.

Curated version of LibriTTS-P with outlier filtering and dynamic prompt generation.
"""

import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torchaudio
from datasets import Dataset as HFDataset
from torch import Tensor
from torch.utils.data import Dataset

from .libritts_p3 import (
    download_libritts_audio,
    download_libritts_p_metadata,
    get_libritts_p_metadata_paths,
    load_libritts_p_prompts,
)


class LIBRITTS_P_Custom(Dataset):
    """Custom LibriTTS-P dataset with outlier filtering and dynamic prompt generation.

    This dataset merges curated quality labels (Z-scores) with original LibriTTS-P metadata.
    It provides a unified interface for loading audio and generating mixed prompts
    deterministically based on the current training epoch.

    Args:
        root (str or Path): Root directory of the dataset.
        url (str, optional): Dataset split. Allowed values: ``"dev-clean"``, 
            ``"dev-other"``, ``"test-clean"``, ``"test-other"``, ``"train-clean-100"``, 
            ``"train-clean-360"``, ``"train-other-500"``.
            (default: ``"train-clean-100"``)
        annotator (str, optional): Speaker prompt annotator. Must be one of 
            ``["df1", "df2", "df3"]``. (default: ``"df1"``)
        max_z_score (float, optional): Maximum modified Z-score for outlier filtering.
            Samples with distance Z-scores above this threshold are excluded. 
            (default: ``3.5``)
        min_group_size (int, optional): Minimum number of samples required per speaker group.
            Groups with fewer samples than this threshold are excluded.
            (default: ``10``)
        use_generated_prompt (bool, optional): If True, uses self-generated style and
            speaker prompts instead of the original LibriTTS-P dataset prompts.
            (default: ``False``)
        transform (Callable, optional): A function/transform that takes in a sample
            dictionary and returns a transformed version. (default: ``None``)
        download (bool, optional): Whether to download the dataset if not found at root.
            (default: ``False``)
        force_reload (bool, optional): If True, re-creates the cache even if it exists.
            (default: ``False``)
    """
    def __init__(
        self,
        root: Union[str, Path],
        url: str = "train-clean-100",
        annotator: str = "df1",
        max_z_score: float = 3.5,
        min_group_size: int = 0,
        use_generated_prompt: bool = False,
        transform: Optional[Callable] = None,
        download: bool = False,
        force_reload: bool = False,
    ) -> None:
        self.root = Path(root)
        self.url = url
        self.annotator = annotator
        self.max_z_score = max_z_score
        self.min_group_size = min_group_size
        self.use_generated_prompt = use_generated_prompt
        self.transform = transform
        self.download = download
        self.epoch = 0

        self.cache_dir = self.root / ".cache" / f"libritts_p_{url}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_source = self._load_or_create_dataset(force_reload)
        if self.use_generated_prompt:
            self.style_map, self.speaker_map = self._load_generated_prompts()
        else:
            self.style_map, self.speaker_map = load_libritts_p_prompts(self.root, self.annotator)
        
        self._filter_outliers()
        self._filter_min_group_size()

    def _load_generated_prompts(self):
        """Loads generated style and speaker prompts from LibriTTS_P directory.

        Returns:
            Tuple:
                - Dict[str, Tuple[List[str], str]]: Mapping from utterance ID to
                  (generated style prompts, style_key).
                - Dict[Tuple[int, str], List[str]]: Mapping from (speaker_id, style_key)
                  to generated speaker descriptions.
        """
        libritts_p_dir = self.root / "LibriTTS_P"

        style_gen_path = libritts_p_dir / "style_prompts_generated.csv"
        style_gen_df = pd.read_csv(
            style_gen_path, sep='|', header=None, names=['style_key', 'description']
        )
        style_by_key = {
            row.style_key: [row.description]
            for row in style_gen_df.itertuples(index=False)
        }

        paths = get_libritts_p_metadata_paths(self.root)
        metadata_df = pd.read_csv(paths["metadata"])
        style_map = {
            row.item_name: (style_by_key.get(row.style_prompt_key, ["Unknown Style"]), row.style_prompt_key)
            for row in metadata_df.itertuples(index=False)
        }

        gen_spk_path = libritts_p_dir / f"generated_{self.annotator}.csv"
        gen_spk_df = pd.read_csv(
            gen_spk_path, sep='|', header=None, names=['speaker_id', 'style_key', 'description']
        )
        speaker_map = {}
        for row in gen_spk_df.itertuples(index=False):
            key = (int(row.speaker_id), row.style_key)
            speaker_map.setdefault(key, []).append(row.description)

        return style_map, speaker_map

    def _load_or_create_dataset(self, force_reload: bool) -> HFDataset:
        """Loads the dataset from disk or triggers creation if missing.

        Args:
            force_reload (bool): If True, forces re-creation of the cache.

        Returns:
            HFDataset: The loaded or newly created Hugging Face dataset.
        """
        cache_path = self.cache_dir / "dataset"
        
        if not force_reload and cache_path.exists():
            try:
                print(f"[INFO] Loading cached dataset from {cache_path}...")
                return HFDataset.load_from_disk(str(cache_path))
            except Exception as e:
                print(f"[INFO] Failed to load cache: {e}. Recreating...")
        
        return self._create_and_cache_dataset(cache_path)

    @staticmethod
    def add_file_metadata(batch: Dict[str, Any], root: Path, url: str) -> Dict[str, List[Any]]:
        """Maps utterance IDs to absolute file system paths and loads normalized text.

        Args:
            batch (Dict): A batch of samples from the Arrow dataset.
            root (Path): Root directory for audio and text resolution.
            url (str): Dataset split URL.

        Returns:
            Dict: A dictionary containing the lists of resolved audio paths and normalized texts.
        """
        audio_paths = []
        normalized_texts = []
        
        for fileid in batch['utterance_id']:
            parts = fileid.split('_')
            spk_id, ch_id = parts[0], parts[1]
            
            # Resolve audio path
            audio_p = root / "LibriTTS" / url / spk_id / ch_id / f"{fileid}.wav"
            audio_paths.append(str(audio_p))
            
            # Resolve and load normalized text
            norm_p = root / "LibriTTS" / url / spk_id / ch_id / f"{fileid}.normalized.txt"
            try:
                with open(norm_p, 'r', encoding='utf-8') as f:
                    normalized_texts.append(f.read().strip())
            except Exception:
                normalized_texts.append("")
                
        return {
            "audio_path": audio_paths,
            "normalized_text": normalized_texts
        }

    def _create_and_cache_dataset(self, cache_path: Path) -> HFDataset:
        """Processes raw CSVs and merges them into a unified Arrow dataset.

        Args:
            cache_path (Path): Destination for the saved Arrow dataset.

        Returns:
            HFDataset: The processed Hugging Face dataset.
        """
        print("[INFO] Creating dataset cache... This may take a while.")
        
        if self.download:
            download_libritts_audio(self.root, self.url)
            download_libritts_p_metadata(self.root, self.annotator)
        
        url_norm = self.url.replace("-", "_")
        curated_filename = f"libritts_p_curated_{url_norm}.csv"
        curated_path = self.root / curated_filename
        if not curated_path.exists():
            raise FileNotFoundError(f"Curated metadata file '{curated_filename}' not found.")
        curated_df = pd.read_csv(curated_path)
        
        paths = get_libritts_p_metadata_paths(self.root)
        orig_meta_df = pd.read_csv(paths["metadata"])
        
        print("[INFO] Merging curated and original metadata...")
        df = pd.merge(
            curated_df, 
            orig_meta_df, 
            left_on="utterance_id", 
            right_on="item_name",
            how="inner",
        )
        
        # Standardize external source IDs and text fields to our local schema
        df = df.rename(columns={
            'spk_id': 'speaker_id',
            'content_prompt': 'original_text'
        })

        dataset = HFDataset.from_pandas(df)
        
        print("[INFO] Mapping audio paths and loading text metadata...")
        dataset = dataset.map(
            self.add_file_metadata, 
            batched=True,
            fn_kwargs={"root": self.root, "url": self.url}
        )
        
        cols_to_remove = [
            c for c in ["item_name", "chapter_id", "style_prompt_key", "__index_level_0__"] 
            if c in dataset.column_names
        ]
        dataset = dataset.remove_columns(cols_to_remove)

        print(f"[INFO] Saving dataset cache to {cache_path}...")
        dataset.save_to_disk(str(cache_path))
        return dataset

    def _filter_outliers(self) -> None:
        """Excludes samples with modified Z-scores exceeding the defined threshold.
        
        Returns:
            None
        """
        if self.max_z_score is None:
            self.data = self.data_source
            return

        print(f"[INFO] Filtering outliers (max_z_score={self.max_z_score})...")
        self.data = self.data_source.filter(
            lambda x: x['distance_z_score'] <= self.max_z_score if x['distance_z_score'] is not None else True
        )
        print(f"[INFO] Filtered: {len(self.data_source)} -> {len(self.data)} samples.")

    def _filter_min_group_size(self) -> None:
        """Excludes samples whose speaker groups have fewer than min_group_size samples.

        Returns:
            None
        """
        if self.min_group_size is None:
            return

        before = len(self.data)
        print(f"[INFO] Filtering groups with fewer than {self.min_group_size} samples...")
        self.data = self.data.filter(
            lambda x: x['group_size'] >= self.min_group_size
        )
        print(f"[INFO] Filtered: {before} -> {len(self.data)} samples.")

    def set_epoch(self, epoch: int) -> None:
        """Sets the current epoch to ensure deterministic variety in prompt mixing.

        Args:
            epoch (int): The current training epoch number.

        Returns:
            None
        """
        self.epoch = epoch

    def _load_audio(self, path: str) -> Tuple[Tensor, int]:
        """Loads an audio file from the filesystem.

        Args:
            path (str): The file path to the audio.

        Returns:
            Tuple[Tensor, int]: A tuple containing the waveform tensor and sample rate.
        """
        waveform, sample_rate = torchaudio.load(path)
        return waveform, sample_rate

    def _mix_prompts(self, style_prompts: List[str], speaker_prompts: List[str], idx: int) -> str:
        """Combines style and speaker prompts into a single descriptive string.

        Args:
            style_prompts (List[str]): List of available style prompt variants.
            speaker_prompts (List[str]): List of available speaker trait variants.
            idx (int): Global index of the sample for deterministic selection.

        Returns:
            str: The final combined prompt.
        """
        seed = int(self.epoch * 1e6 + idx)
        rng = random.Random(seed)
        
        style = rng.choice(style_prompts)
        
        spk_prompts_copy = list(speaker_prompts).copy()
        rng.shuffle(spk_prompts_copy)
        spk_str = ", ".join(spk_prompts_copy)
        
        return f"{style} The speaker's identity can be described as {spk_str}."

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns the n-th sample from the dataset.

        Args:
            idx (int): The index of the sample to load.

        Returns:
            Dict: A dictionary containing:
                - waveform (Tensor): Audio signal.
                - sample_rate (int): Audio sample rate.
                - original_text (str): Ground truth text.
                - normalized_text (str): Normalized ground truth text.
                - speaker_id (int): ID of the speaker.
                - utterance_id (str): ID of the utterance.
                - group_id (str): ID of the prompt group.
                - group_size (int): Size of the group.
                - distance_z_score (float): Outlier score based on embedding distance.
                - intra_group_distance (float): Average distance within the group.
                - combined_prompt (str): Deterministically mixed description.
                - style_prompts (List[str]): Raw list of style prompt variants.
                - speaker_prompts (List[str]): Raw list of speaker trait variants.
        """
        item = self.data[idx]
        
        waveform, sr = self._load_audio(item['audio_path'])
        
        style_variants, style_key_or_spk = self.style_map.get(item['utterance_id'], (["Unknown Style"], None))

        speaker_id = int(item['speaker_id'])
        if self.use_generated_prompt:
            speaker_variants = self.speaker_map.get((speaker_id, style_key_or_spk), ["Unknown Speaker"])
        else:
            speaker_variants = self.speaker_map.get(speaker_id, ["Unknown Speaker"]) if self.speaker_map else ["No Annotator"]
        
        combined_prompt = self._mix_prompts(
            style_variants, 
            speaker_variants, 
            idx,
        )
        
        return {
            'waveform': waveform,
            'sample_rate': sr,
            'original_text': item.get('original_text', ''),
            'normalized_text': item.get('normalized_text', ''),
            'style_prompts': style_variants,
            'speaker_prompts': speaker_variants,
            'combined_prompt': combined_prompt,
            'utterance_id': item['utterance_id'],
            'speaker_id': speaker_id,
            'group_id': item.get('group_id', ''),
            'group_size': item.get('group_size', 0),
            'distance_z_score': item.get('distance_z_score', 0.0),
            'intra_group_distance': item.get('intra_group_distance', 0.0)
        }
