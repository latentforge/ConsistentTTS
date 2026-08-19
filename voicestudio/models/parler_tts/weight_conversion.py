"""Checkpoint conversion for Parler-TTS."""

import re

from transformers.core_model_loading import WeightRenaming


def _res_unit_renaming(old_prefix: str, new_prefix: str, block_offset: int) -> list[WeightRenaming]:
    rules = []
    for unit_idx in range(3):
        old_unit = rf"{re.escape(old_prefix)}\.{unit_idx + block_offset}\.block"
        new_unit = f"{new_prefix}.res_unit{unit_idx + 1}"
        rules.append(WeightRenaming(rf"{old_unit}\.1\.(.*)", rf"{new_unit}.conv1.\1"))
        rules.append(WeightRenaming(rf"{old_unit}\.3\.(.*)", rf"{new_unit}.conv2.\1"))
        rules.append(WeightRenaming(rf"{old_unit}\.0\.alpha", rf"{new_unit}.snake1.alpha"))
        rules.append(WeightRenaming(rf"{old_unit}\.2\.alpha", rf"{new_unit}.snake2.alpha"))
    return rules


def build_dac_weight_conversion_mapping(prefix: str = "audio_encoder.") -> list[WeightRenaming]:
    """
    Builds the `WeightRenaming` rules that translate a `descript-audio-codec`-style state dict (the format
    original Parler-TTS checkpoints ship their audio encoder weights in) into the module layout of the
    `transformers` `DacModel`.

    Args:
        prefix (`str`, *optional*, defaults to `"audio_encoder."`):
            Prefix under which the audio encoder submodule's weights live in the composite Parler-TTS
            checkpoint's state dict.

    Returns:
        `list[WeightRenaming]`: Rules to pass to the model loader's weight conversion mapping.
    """
    p = re.escape(prefix)
    rules = [
        WeightRenaming(rf"{p}model\.encoder\.block\.0\.(.*)", rf"{prefix}encoder.conv1.\1"),
        WeightRenaming(rf"{p}model\.encoder\.block\.5\.alpha", rf"{prefix}encoder.snake1.alpha"),
        WeightRenaming(rf"{p}model\.encoder\.block\.6\.(.*)", rf"{prefix}encoder.conv2.\1"),
        WeightRenaming(rf"{p}model\.decoder\.model\.0\.(.*)", rf"{prefix}decoder.conv1.\1"),
        WeightRenaming(rf"{p}model\.decoder\.model\.5\.alpha", rf"{prefix}decoder.snake1.alpha"),
        WeightRenaming(rf"{p}model\.decoder\.model\.6\.(.*)", rf"{prefix}decoder.conv2.\1"),
        WeightRenaming(
            rf"{p}model\.quantizer\.quantizers\.(\d+)\.codebook\.weight",
            rf"{prefix}quantizer.quantizers.\1.codebook.weight",
        ),
        WeightRenaming(
            rf"{p}model\.quantizer\.quantizers\.(\d+)\.(in|out)_proj\.(.*)",
            rf"{prefix}quantizer.quantizers.\1.\2_proj.\3",
        ),
    ]
    for i in range(4):
        rules += _res_unit_renaming(f"{prefix}model.encoder.block.{i + 1}.block", f"{prefix}encoder.block.{i}", 0)
        rules.append(
            WeightRenaming(rf"{p}model\.encoder\.block\.{i + 1}\.block\.3\.alpha", rf"{prefix}encoder.block.{i}.snake1.alpha")
        )
        rules.append(
            WeightRenaming(
                rf"{p}model\.encoder\.block\.{i + 1}\.block\.4\.(.*)", rf"{prefix}encoder.block.{i}.conv1.\1"
            )
        )
        rules.append(
            WeightRenaming(
                rf"{p}model\.decoder\.model\.{i + 1}\.block\.0\.alpha", rf"{prefix}decoder.block.{i}.snake1.alpha"
            )
        )
        rules.append(
            WeightRenaming(
                rf"{p}model\.decoder\.model\.{i + 1}\.block\.1\.(.*)", rf"{prefix}decoder.block.{i}.conv_t1.\1"
            )
        )
        rules += _res_unit_renaming(f"{prefix}model.decoder.model.{i + 1}.block", f"{prefix}decoder.block.{i}", 2)
    return rules


def convert_parler_tts_state_dict(state_dict: dict, prefix: str = "audio_encoder.") -> dict:
    """
    Applies [`build_dac_weight_conversion_mapping`] to a Parler-TTS checkpoint's state dict, translating its
    `descript-audio-codec`-style audio encoder weights into the `transformers` `DacModel` layout. Keys outside
    the audio encoder are passed through unchanged.

    Args:
        state_dict (`dict`):
            The state dict of a Parler-TTS checkpoint saved against the original `descript-audio-codec` layout.
        prefix (`str`, *optional*, defaults to `"audio_encoder."`):
            Prefix under which the audio encoder submodule's weights live in `state_dict`.

    Returns:
        `dict`: A state dict with the audio encoder weights renamed to the `DacModel` layout.
    """
    rules = build_dac_weight_conversion_mapping(prefix=prefix)
    converted = {}
    for key, value in state_dict.items():
        renamed_key = key
        for rule in rules:
            candidate, matched = rule.rename_source_key(key)
            if matched is not None:
                renamed_key = candidate
                break
        converted[renamed_key] = value
    return converted
