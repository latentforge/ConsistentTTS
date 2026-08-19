"""UTMOSv2 naturalness MOS metric."""

from typing import List, Optional, Union

import numpy as np

import evaluate


_CITATION = """\
@inproceedings{baba2024utmosv2,
    title = {The T05 System for The VoiceMOS Challenge 2024: Transfer Learning from Deep Image Classifier to Naturalness MOS Prediction of High-Quality Synthetic Speech},
    author = {Baba, Kaito and Nakata, Wataru and Saito, Yuki and Saruwatari, Hiroshi},
    booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
    year = {2024},
}
"""

_DESCRIPTION = """\
UTMOSv2 predicts the naturalness Mean Opinion Score (MOS) of synthesized speech without a
reference recording. This metric wraps the pretrained ensemble released by the UTMOSv2 authors
(https://github.com/sarulab-speech/UTMOSv2) behind the standard `evaluate.Metric` interface, so
callers depend on `evaluate` rather than importing the upstream training/eval repository directly.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of speech file paths, or a list of waveform arrays paired with `sampling_rate`.
    sampling_rate (`int`, *optional*): sampling rate of the waveform arrays in `predictions`. Required
        when `predictions` are arrays rather than file paths.
    device (`str`, *optional*): device to run inference on, e.g. `"cuda"` or `"cpu"`. Defaults to the
        device UTMOSv2 selects automatically.

Returns:
    mos (`List[float]`): predicted naturalness MOS, one value per prediction, on the 1-5 scale.
"""


class UTMOSv2Metric(evaluate.Metric):
    """
    Reference-free naturalness MOS metric backed by the pretrained UTMOSv2 ensemble.

    Loading the pretrained weights requires the `utmosv2` package (`pip install
    git+https://github.com/sarulab-speech/UTMOSv2`). It is not a VoiceStudio dependency: UTMOSv2's
    architecture is a five-fold ensemble that fuses an SSL branch with an image-classifier branch
    over mel-spectrogram crops, driven by a hydra config system for architecture/fold selection.
    That is real model code, not a thin scoring call, so it is not vendored here; this class only
    adapts the upstream package's public `create_model`/`predict` API to the `evaluate.Metric`
    shape and lazily imports it so `voicestudio` itself never requires it at import time.
    """

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=evaluate.Features(
                {
                    "predictions": evaluate.Value("string"),
                }
            ),
        )

    def _download_and_prepare(self, dl_manager):
        try:
            import utmosv2
        except ImportError as e:
            raise ImportError(
                "UTMOSv2Metric requires the `utmosv2` package: "
                "pip install git+https://github.com/sarulab-speech/UTMOSv2"
            ) from e

        self._model = utmosv2.create_model(pretrained=True)

    def _compute(
        self,
        predictions: List[Union[str, np.ndarray]],
        sampling_rate: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if device is not None:
            self._model = self._model.to(device)

        mos = []
        for prediction in predictions:
            if isinstance(prediction, str):
                score = self._model.predict(input_path=prediction)
            else:
                if sampling_rate is None:
                    raise ValueError(
                        "`sampling_rate` is required when `predictions` are waveform arrays."
                    )
                score = self._model.predict(data=prediction, sr=sampling_rate)
            mos.append(float(score))

        return {"mos": mos}
