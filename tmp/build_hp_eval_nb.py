"""Build two eval notebooks (GPU1/GPU2) from the validated kablation harness to evaluate
the HP-sweep VQ-query checkpoints (method2 COS/WER/UTMOS). Each writes results/_summary/<tag>.json."""
import json, sys

G1 = [("lr005_s1","ckpt/hp/lr005_s1.pt"),("lr01_s1","ckpt/hp/lr01_s1.pt"),("lr04_s1","ckpt/hp/lr04_s1.pt"),
      ("const_s1","ckpt/hp/const_s1.pt"),("lin_s1","ckpt/hp/lin_s1.pt"),("wu30_s1","ckpt/hp/wu30_s1.pt")]
G2 = [("lr005_s2","ckpt/hp/lr005_s2.pt"),("lr01_s2","ckpt/hp/lr01_s2.pt"),("lr04_s2","ckpt/hp/lr04_s2.pt"),
      ("const_s2","ckpt/hp/const_s2.pt"),("lin_s2","ckpt/hp/lin_s2.pt"),("wu30_s2","ckpt/hp/wu30_s2.pt")]

def build(gpu, items, outname):
    j = json.load(open("experiment_qwen_vqr_kablation.ipynb"))
    j["cells"][2]["source"] = ['import os\n', f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu}"\n']
    lst = ",".join(f"('{t}','{p}')" for t, p in items)
    j["cells"][26]["source"] = [
        "import json as _json\n",
        f"HP = [{lst}]\n",
        "os.makedirs('results/_summary', exist_ok=True)\n",
        "for _tag,_path in HP:\n",
        "    if not os.path.exists(_path):\n",
        "        print('SKIP missing',_path,flush=True); continue\n",
        "    _ck = torch.load(_path, map_location=device)\n",
        "    globals()['_QUERY']=_ck['param'].to(device); globals()['_QK']=int(_ck['k'])\n",
        "    globals()['REF_TEXT']=tokenizer.decode(_ck['token_ids'])\n",
        "    OUTPUT_DIR=f'./results/{model_id}_{_tag}'; os.makedirs(OUTPUT_DIR,exist_ok=True)\n",
        "    print(f'===== {_tag} =====',flush=True)\n",
        "    _res=save_and_evaluate(model,OUTPUT_DIR,disable_save=True)\n",
        "    _m=list(_res.values())[0]\n",
        "    _json.dump({'name':_tag,'metrics':{kk:float(_m[kk]) for kk in ['sim_mean','utmos_mean','wer_mean']}},open(f'results/_summary/{_tag}.json','w'),indent=2)\n",
        "    print(f'[{_tag}] COS={_m[\"sim_mean\"]:.4f} WER={_m[\"wer_mean\"]:.4f} UTMOS={_m[\"utmos_mean\"]:.3f}',flush=True)\n",
    ]
    json.dump(j, open(outname, "w"))
    print("wrote", outname, "for GPU", gpu, "with", len(items), "ckpts")

build(1, G1, "new_hp_eval_g1.ipynb")
build(2, G2, "new_hp_eval_g2.ipynb")
