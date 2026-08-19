"""Copy of experiment_qwen.ipynb's OFFICIAL eval harness (EvaluationPipeline + strategy),
but with the VC model = Qwen3-TTS Base and a voice-cloning synthesize (x-vector clone
from the reference audio the harness already passes). Reports official method1 (SECS,
cos-to-reference) and method2 (Gen-Gen) SIM. GPU 1 only."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# VC 모델을 캐노니컬 하네스에 넣어 공식 성능 측정
`experiment_qwen.ipynb`의 **공식 평가 골격(cell 46–57: EvaluationPipeline + create_strategy)을 그대로** 사용.
바꾼 것은 두 가지뿐: (1) 모델 = `Qwen3-TTS-12Hz-1.7B-Base`(VC), (2) `TestModel.synthesize`가
harness가 넘겨주는 `reference_audio`로 **음향 클로닝**(x-vector). method1(참조 대비 SECS)·method2(Gen-Gen) 모두 측정.""")

code("""import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings; warnings.filterwarnings("ignore")
import torch, numpy as np, soundfile as sf
from pathlib import Path
device_map = "cuda:0"
print("cuda:", torch.cuda.get_device_name(0))""")

# --- model load: Base VC via wrapper ---
code("""# Model select = VC (Base)
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
import transformers; transformers.logging.set_verbosity_error()
from voicestudio._qwen3_tts.inference.qwen3_tts_model import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained(model_id, device_map=device_map,
            dtype=torch.bfloat16, attn_implementation="flash_attention_2")
if getattr(model.model, "speech_tokenizer", None) is None:
    from voicestudio._qwen3_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
    from transformers.utils import cached_file
    cf = cached_file(model_id, "speech_tokenizer/preprocessor_config.json")
    model.model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(os.path.dirname(cf)))
print("VC Base loaded; speech_tokenizer:", model.model.speech_tokenizer is not None)""")

# --- canonical eval harness cells (verbatim 46,47,48) ---
code("""from spk_incon.metrics.presets import DatasetType, GenerationMethod, SynthesisConfig, ModelType
from spk_incon.metrics.strategies import create_strategy
from spk_incon.datasets import DatasetType, create_dataset

from spk_incon.utils.evaluate import EvaluationPipeline""")
code("""test_config = SynthesisConfig()
test_dataset_type = DatasetType.LIBRITTS
test_dataset_config = test_config.get_dataset_config(test_dataset_type.value)""")
code("""test_dataset = create_dataset(test_dataset_type, test_dataset_config, root_dir="./data")""")

# --- OUTPUT_DIR (canonical cell 38) ---
code("""OUTPUT_DIR = "./results/" + model_id + "_VC_canonical"
os.makedirs(OUTPUT_DIR, exist_ok=True)""")

# --- TestModel with VC synthesize (x-vector clone from reference_audio) ---
code("""import random
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class TestModel:
    @classmethod
    def seed_everything(cls, seed: int = 42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    @classmethod
    def synthesize(cls, text, output_path, reference_audio=None, style_prompt=None, speaker_id=None) -> bool:
        cls.seed_everything()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # VC: clone the speaker identity from the reference audio the harness passes.
        rw, rsr = sf.read(str(reference_audio)); rw = np.asarray(rw, dtype=np.float32)
        if rw.ndim > 1: rw = rw.mean(1)
        prompt = model.create_voice_clone_prompt(ref_audio=(rw, rsr), x_vector_only_mode=True)
        wavs, sr = model.generate_voice_clone(text=[text], voice_clone_prompt=prompt,
                                              do_sample=True, top_k=50, temperature=0.9)
        sf.write(output_path, np.asarray(wavs[0]), sr)
        try: return output_path.stat().st_size > 0
        except FileNotFoundError: return False""")

# --- canonical 50, 51 (ModelType, evaluator) ---
code("""from enum import Enum
class ModelType(Enum):
    TEST = model.__class__.__name__""")
code("""test_model_type = ModelType.TEST
test_model = TestModel()
evaluator = EvaluationPipeline(base_dir=Path(OUTPUT_DIR+"_last"))
test_config.generation.output_dir = Path(OUTPUT_DIR+"_last")
print("model_type:", test_model_type.value)""")

# --- METHOD2 (canonical 53,54) ---
md("""## method2 — Gen-Gen 자기일관성 (syn₀ vs synᵢ)""")
code("""strategy = create_strategy(GenerationMethod.METHOD2, test_config, test_dataset, test_model)
exp2_result = strategy.generate_all(test_dataset_type.value, test_model_type.value)
print("method2 gen done:", exp2_result)""")
code("""exp2_eval_result = evaluator.evaluate_dataset_model(
    dataset_type=test_dataset_type, model_type=test_model_type, methods=[GenerationMethod.METHOD2])
m2 = list(exp2_eval_result.values())[0]
print(f"[VC method2] COS(sim_mean)={m2['sim_mean']:.4f}  WER={m2.get('wer_mean',float('nan')):.4f}  UTMOS={m2.get('utmos_mean',float('nan')):.3f}")""")

# --- METHOD1 (canonical 56,57) ---
md("""## method1 — 참조 대비 SECS (cos(syn, 실제 참조음성)) — VC의 표준 지표(≈0.6 후보)""")
code("""strategy1 = create_strategy(GenerationMethod.METHOD1, test_config, test_dataset, test_model)
exp1_result = strategy1.generate_all(test_dataset_type.value, test_model_type.value)
print("method1 gen done:", exp1_result)""")
code("""exp1_eval_result = evaluator.evaluate_dataset_model(
    dataset_type=test_dataset_type, model_type=test_model_type, methods=[GenerationMethod.METHOD1])
m1 = list(exp1_eval_result.values())[0]
print(f"[VC method1] SECS(sim_mean)={m1['sim_mean']:.4f}  WER={m1.get('wer_mean',float('nan')):.4f}")""")

code("""print("==== VC (Base) via canonical harness ====")
print(f"  method1 (cos to reference / SECS) : {m1['sim_mean']:.4f}")
print(f"  method2 (Gen-Gen consistency)    : {m2['sim_mean']:.4f}")""")

nb["cells"]=cells
nbf.write(nb, "new_vc_canonical.ipynb")
print("wrote new_vc_canonical.ipynb with", len(cells), "cells")
