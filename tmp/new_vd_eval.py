import os, sys, json
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
from enum import Enum
from pathlib import Path
from spk_incon.metrics.presets import DatasetType, GenerationMethod
from spk_incon.metrics import MetricType
from spk_incon.utils.evaluate import EvaluationPipeline
tag=sys.argv[1]
MT=Enum("MT",{"TEST":"Qwen3TTSForConditionalGeneration"})
base=Path(f"results/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign_{tag}")
ev=EvaluationPipeline(base_dir=base,html=False,verbose=False)
r=ev.evaluate_dataset_model(DatasetType.LIBRITTS,MT.TEST,metric_types=[MetricType.UTMOS,MetricType.WER,MetricType.SIM],methods=[GenerationMethod.METHOD2])
m=list(r.values())[0]
out={k:float(m[k]) for k in ["sim_mean","wer_mean","utmos_mean","sim_std","sim_median"]}
json.dump({"name":tag,"metrics":out},open(f"results/_summary/{tag}.json","w"),indent=2)
print(f"[{tag}] COS={m['sim_mean']:.4f} WER={m['wer_mean']:.4f} UTMOS={m['utmos_mean']:.3f}")
