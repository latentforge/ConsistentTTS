# PromptTTS++

PromptTTS++ synthesizes speech from text where speaker identity and speaking style are controlled by a
natural-language prompt. This implementation is built on top of `transformers-tts`'s `FastSpeech2Conformer`
(phoneme encoder, variance adaptor, conformer decoder) for the acoustic model and `FastSpeech2ConformerHifiGan` for
the vocoder. The prompt encoder, which turns the natural-language style prompt into a style embedding through a
BERT model and an adaptor MLP, is implemented directly in this folder since it has no equivalent already in
`transformers-tts`.

Original model and code: https://github.com/line/promptttspp
