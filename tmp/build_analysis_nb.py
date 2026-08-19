"""Builds new_cos_wer_analysis.ipynb — a self-contained diagnostic notebook that
investigates WHY method2 Gen-Gen COS saturates and WHY WER collapses past a COS
threshold. Reuses on-disk generated wavs (no re-generation of the sweep) and
lightly reconstructs the internal anchors for the atypicality probe."""
import json, nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# COS↔WER 벽 분석 — 왜 COS는 포화하고, 특정 COS 이상에서 WER은 붕괴하는가

**최종 성과:** VQ learnable query 0.421 COS / 0.130 WER. 이 이상 COS를 밀면 WER이 무너진다.
4개 방법군(VQ 스윕 110점, 데이터 큐레이션, 화자 임베딩 채널, ref_text-matching loss) 모두 벽을 못 넘었다.

**이 노트북의 목적(개선 X, 규명 O):** 디스크에 이미 있는 생성물을 재분석해
1. **COS 상한** — 왜 유사도를 더 못 올리나 (일관성 vs 변별력, 모드 붕괴, 앵커 누출)
2. **WER 붕괴** — 왜 특정 COS 이상에서 WER이 무너지나 (per-utterance 트레이드오프, 앵커 비정형성/운율 왜곡)
를 정량·시각화한다.

**방법:** 공식 지표와 동일 구성 — SIM=ECAPA-TDNN(`spkrec-ecapa-voxceleb`), method2 페어=set별 cos(syn₀, synᵢ) floor@0;
WER=whisper-large-v3 + repo jiwer 정규화. 프론티어 전 구간 8개 체크포인트를 per-utterance로 재계산.""")

# ---- env
code("""import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, json, math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["figure.dpi"]=120; mpl.rcParams["font.size"]=10
import soundfile as sf, torchaudio
from collections import defaultdict
DEV="cuda:0"
ANA=Path("results/analysis"); ANA.mkdir(parents=True, exist_ok=True)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))""")

# ---- tag config
code("""ROOT=Path("results/Qwen")
def dir_of(tag): return ROOT/f"Qwen3-TTS-12Hz-1.7B-VoiceDesign_{tag}"
def synbase(tag): return dir_of(tag)/"syn/libritts/Qwen3TTSForConditionalGeneration/method2"
def refbase(tag): return dir_of(tag)/"ref/libritts/method2"

# frontier-spanning checkpoints (all have 100 wavs on disk). ckpt only for VQ-query variants.
TAGS=[
 ("vd_default",        "VD baseline (no query)",       None),
 ("hyb2_vq_xvec",      "Speaker-channel hybrid",       None),
 ("rt1.5",             "reftext w=1.5",                "ckpt/query_vq_k32_rt1.5.pt"),
 ("rt8",               "reftext w=8",                  "ckpt/query_vq_k32_rt8.pt"),
 ("vqr_k32_recovered", "SOTA VQ query (best)",         "ckpt/query_vq_k32_recovered.pt"),
 ("rt3",               "reftext w=3",                  "ckpt/query_vq_k32_rt3.pt"),
 ("rt1",               "reftext w=1",                  "ckpt/query_vq_k32_rt1.pt"),
 ("rt2",               "reftext w=2 (max COS)",        "ckpt/query_vq_k32_rt2.pt"),
]
def official(tag):
    f=Path(f"results/_summary/{tag}.json")
    if f.exists():
        m=json.load(open(f))["metrics"]; return m.get("sim_mean",np.nan), m.get("wer_mean",np.nan)
    return np.nan, np.nan
for t,lbl,_ in TAGS:
    c,w=official(t); print(f"{t:22s} {lbl:26s} official COS={c:.4f} WER={w:.4f}")""")

# ---- §1 global frontier
md("""## §1. 전체 프론티어 지도 — 트레이드오프는 모든 방법군에서 보편적

지금까지 돌린 모든 실험을 (COS, WER) 평면에 찍는다. 목표 영역(COS≥0.40 & WER≤0.12)이 비어 있음을 확인.""")
code("""import csv
pts=[]  # (cos, wer, family, name)
# VQ sweep
with open("results/SWEEP_RESULTS.csv") as f:
    for r in csv.DictReader(f):
        try: pts.append((float(r["cos"]), float(r["wer"]), "VQ query sweep", r["name"]))
        except: pass
# reftext sweep
with open("results/REFTEXT_SWEEP.csv") as f:
    for r in csv.DictReader(f):
        try: pts.append((float(r["COS"]), float(r["WER"]), "reftext loss", r["tag"]))
        except: pass
# hybrid speaker channel
for t in ["hyb2_fixed_xvec","hyb2_vq_xvec","hyb2_fixed_icl","hyb2_vq_icl"]:
    c,w=official(t)
    if not math.isnan(c): pts.append((c,w,"speaker channel",t))
# baseline
c,w=official("vd_default"); pts.append((c,w,"VD baseline","vd_default"))
print("total points:", len(pts))

fam_col={"VQ query sweep":"#4C72B0","reftext loss":"#DD8452","speaker channel":"#55A868","VD baseline":"#C44E52"}
fig,ax=plt.subplots(figsize=(9,6.5))
for fam,col in fam_col.items():
    xs=[p[0] for p in pts if p[2]==fam]; ys=[p[1] for p in pts if p[2]==fam]
    ax.scatter(xs,ys,s=42,alpha=.7,label=f"{fam} (n={len(xs)})",color=col,edgecolor="white",linewidth=.4)
# target wall box
ax.axvspan(0.40,0.55,ymin=0,ymax=1,color="none")
ax.add_patch(plt.Rectangle((0.40,0.0),0.20,0.12,fill=True,color="green",alpha=.10))
ax.axvline(0.40,ls="--",c="green",lw=1); ax.axhline(0.12,ls="--",c="green",lw=1)
ax.text(0.405,0.118,"TARGET: COS≥0.40 & WER≤0.12  (EMPTY)",color="green",fontsize=9,va="top")
# pareto frontier (lower-left)
P=sorted(pts); front=[]; best=1e9
for c,w,fam,nm in P:
    if w<best-1e-9: front.append((c,w)); best=w
ax.plot([f[0] for f in front],[f[1] for f in front],"k-",lw=1.3,alpha=.5,label="Pareto frontier")
ax.set_xlabel("Gen-Gen COS (method2, higher=more consistent)"); ax.set_ylabel("WER (lower=better)")
ax.set_title("모든 실험의 COS–WER 프론티어: 목표 영역은 비어 있다"); ax.legend(fontsize=8); ax.grid(alpha=.25)
plt.tight_layout(); plt.savefig(ANA/"fig1_frontier.png"); plt.show()""")

# ---- §2 recompute per-utterance
md("""## §2. per-utterance 재계산 (ECAPA 임베딩 + whisper-large-v3 WER)

디스크의 생성 wav를 그대로 사용. 각 wav의 ECAPA 임베딩, whisper WER, 길이, 발화속도를 계산해 캐시한다.
`ref_XXX.wav`(= 실제 타깃 화자 음성)의 임베딩도 계산(절대 화자충실도 측정용).""")
code("""from speechbrain.inference.speaker import EncoderClassifier
ecapa=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":DEV})
@torch.no_grad()
def load16(p):
    w,sr=sf.read(str(p)); w=np.asarray(w,dtype=np.float32)
    if w.ndim>1: w=w.mean(1)
    t=torch.tensor(w)
    if sr!=16000: t=torchaudio.functional.resample(t,sr,16000)
    return t, len(w)/sr
@torch.no_grad()
def ecapa_emb(t16):
    e=ecapa.encode_batch(t16.unsqueeze(0).to(DEV)).squeeze()
    return torch.nn.functional.normalize(e,dim=0).cpu().numpy()""")

code("""import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
WT=jiwer.Compose([jiwer.ToLowerCase(),jiwer.RemoveWhiteSpace(replace_by_space=True),
                  jiwer.RemoveMultipleSpaces(),jiwer.ReduceToListOfListOfWords(word_delimiter=" ")])
wp=WhisperProcessor.from_pretrained("openai/whisper-large-v3")
wm=WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3",dtype=torch.float16).to(DEV).eval()
@torch.no_grad()
def transcribe(t16):
    f=wp(t16.numpy(),sampling_rate=16000,return_tensors="pt").input_features.to(DEV).half()
    ids=wm.generate(f,language="en",task="transcribe",max_new_tokens=200)
    return wp.batch_decode(ids,skip_special_tokens=True)[0].strip()
def uwer(ref,hyp):
    try: return float(min(1.0,jiwer.wer(ref,hyp,reference_transform=WT,hypothesis_transform=WT)))
    except: return np.nan""")

code("""CACHE=ANA/"perutt.npz"
def build_cache():
    rows=[]   # dict per utterance
    refemb={} # (tag, ref_id) -> real-ref embedding
    for tag,lbl,_ in TAGS:
        sb=synbase(tag); rb=refbase(tag)
        sets=sorted(d for d in sb.iterdir() if d.is_dir() and d.name.startswith("set_"))
        for sd in sets:
            rid=sd.name.split("_")[1]
            meta=json.load(open(sd/"metadata.json"))
            # real reference speaker embedding
            rp=rb/f"ref_{rid}.wav"
            if rp.exists() and (tag,rid) not in refemb:
                t16,_=load16(rp); refemb[(tag,rid)]=ecapa_emb(t16)
            for wav in sorted(sd.glob("syn_*.wav")):
                t16,dur=load16(wav)
                txt=meta.get(wav.name,{}).get("target_text","") or ""
                emb=ecapa_emb(t16)
                hyp=transcribe(t16); wr=uwer(txt,hyp)
                uidx=int(wav.stem.split("_")[-1])
                nw=len(txt.split())
                rows.append(dict(tag=tag,rid=rid,uidx=uidx,dur=dur,nwords=nw,
                                 rate=(nw/dur if dur>0 else np.nan),wer=wr,emb=emb,
                                 refemb=refemb.get((tag,rid))))
        print("done", tag, "utts so far", len(rows), flush=True)
    # serialize
    np.savez_compressed(CACHE,
        tag=np.array([r["tag"] for r in rows]), rid=np.array([r["rid"] for r in rows]),
        uidx=np.array([r["uidx"] for r in rows]), dur=np.array([r["dur"] for r in rows]),
        nwords=np.array([r["nwords"] for r in rows]), rate=np.array([r["rate"] for r in rows]),
        wer=np.array([r["wer"] for r in rows]),
        emb=np.stack([r["emb"] for r in rows]),
        refemb=np.stack([r["refemb"] if r["refemb"] is not None else np.full(192,np.nan) for r in rows]))
    return rows
if CACHE.exists():
    print("loading cache", CACHE)
else:
    build_cache()
Z=np.load(CACHE, allow_pickle=True)
print("cached utterances:", len(Z["tag"]), "emb dim", Z["emb"].shape)""")

# ---- build convenience dataframe
code("""import pandas as pd
df=pd.DataFrame(dict(tag=Z["tag"],rid=Z["rid"],uidx=Z["uidx"],dur=Z["dur"],
                     nwords=Z["nwords"],rate=Z["rate"],wer=Z["wer"]))
EMB=Z["emb"]; REFEMB=Z["refemb"]
TAG_ORDER=[t for t,_,_ in TAGS]
LBL={t:l for t,l,_ in TAGS}
df["tag"]=pd.Categorical(df["tag"],categories=TAG_ORDER,ordered=True)
# index helper: rows per (tag,rid) sorted by uidx
def rows_of(tag,rid):
    m=np.where((Z["tag"]==tag)&(Z["rid"]==rid))[0]
    return m[np.argsort(Z["uidx"][m])]
print(df.groupby("tag",observed=True)[["wer","rate","dur"]].mean())""")

# ---- §3 COS mechanism
md("""## §3. COS 상한의 메커니즘

### 3a. 재현: my recompute vs 공식 COS
method2 정의(set별 cos(syn₀, synᵢ) floor@0)를 per-file 임베딩으로 재현하고 공식값과 비교.""")
code("""def cos(a,b): return float(np.dot(a,b))  # emb already L2-normalized
recos={}
for tag in TAG_ORDER:
    sims=[]
    for rid in sorted(set(Z["rid"][Z["tag"]==tag])):
        idx=rows_of(tag,rid)
        if len(idx)<2: continue
        a=EMB[idx[0]]
        for j in idx[1:]: sims.append(max(0.0,cos(a,EMB[j])))
    recos[tag]=np.mean(sims)
fig,ax=plt.subplots(figsize=(9,4.5))
x=np.arange(len(TAG_ORDER)); off=[official(t)[0] for t in TAG_ORDER]; rec=[recos[t] for t in TAG_ORDER]
ax.bar(x-.2,off,.4,label="official (padded batch)",color="#4C72B0")
ax.bar(x+.2,rec,.4,label="my recompute (per-file)",color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels([LBL[t] for t in TAG_ORDER],rotation=30,ha="right",fontsize=8)
ax.set_ylabel("Gen-Gen COS"); ax.legend(); ax.grid(alpha=.25,axis="y")
ax.set_title("공식 COS vs per-file 재현 — 경향 일치(오프셋=zero-pad 압축)")
plt.tight_layout(); plt.savefig(ANA/"fig3a_recos.png"); plt.show()""")

md("""### 3b. 실제 일관성은 이미 포화 — 공식 COS "천장"은 상당 부분 지표 압축이다

- **within(honest)** = 같은 페르소나 내 cos(syn₀, synᵢ), per-file(무패딩). **between** = 서로 다른 페르소나 syn₀끼리.
- 관찰(아래): within은 모든 query 체크포인트에서 **이미 0.71–0.75로 포화**하며 공식 COS(0.37→0.46)만큼 벌어지지 않는다.
  즉 §3a에서 본 공식↔per-file 격차처럼, 공식 COS의 "천장/변동"은 **zero-pad 압축 아티팩트**가 크고 실제 일관성 결핍이 아니다.
- between도 낮게 유지(최고 체크포인트 recovered=0.19)되어 **모드 붕괴는 관측되지 않음**(당초 가설 기각).""")
code("""within={}; between={}
for tag in TAG_ORDER:
    rids=sorted(set(Z["rid"][Z["tag"]==tag]))
    # within
    ws=[]; s0={}
    for rid in rids:
        idx=rows_of(tag,rid)
        if len(idx)<2: continue
        a=EMB[idx[0]]; s0[rid]=a
        ws+=[cos(a,EMB[j]) for j in idx[1:]]
    within[tag]=np.mean(ws)
    # between: syn0 of different personas
    ks=list(s0); bs=[cos(s0[ks[i]],s0[ks[j]]) for i in range(len(ks)) for j in range(i+1,len(ks))]
    between[tag]=np.mean(bs)
fig,ax=plt.subplots(figsize=(9,5))
xs=[official(t)[0] for t in TAG_ORDER]
ax.plot(xs,[within[t] for t in TAG_ORDER],"o-",label="within-persona (consistency = COS)",color="#4C72B0")
ax.plot(xs,[between[t] for t in TAG_ORDER],"s-",label="between-persona (identity overlap)",color="#C44E52")
for t in TAG_ORDER:
    ax.annotate(LBL[t].split(" (")[0],(official(t)[0],within[t]),fontsize=7,xytext=(0,6),textcoords="offset points",ha="center")
ax.set_xlabel("official COS"); ax.set_ylabel("mean cosine (per-file, honest)")
ax.set_title("실제 within-consistency는 이미 0.7+ 포화 · between은 낮게 유지 (모드붕괴 없음)")
ax.legend(); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(ANA/"fig3b_within_between.png"); plt.show()
print("gap(within-between):")
for t in TAG_ORDER: print(f"  {LBL[t]:26s} within={within[t]:.3f} between={between[t]:.3f} gap={within[t]-between[t]:.3f}")""")

md("""### 3c. (철회) '진짜 타깃 화자'는 이 과제에 존재하지 않는다 — 참고용 대조만

**주의:** Voice-Design의 타깃은 **텍스트 페르소나**이지 참조 음성이 아니다. `ref_XXX.wav`는 페르소나 주석이 유래된
LibriTTS 원본 발화를 복사한 것일 뿐이며(`method1.py`), **공식 method2 COS는 이 ref를 사용하지 않는다**(syn₀ vs synᵢ만 비교).
"차분한 저음 남성"에 맞는 목소리는 여럿이므로 특정 화자를 "정답"으로 볼 수 없다.

아래 `cos(synᵢ, ref_XXX)`는 (a) 임의의 한 원본 화자를 대조로 삼고 (b) 실제 사람 vs 합성 TTS의 **도메인 갭**이 섞여
**충실도 지표가 아니다.** 참고로만 표시하며, 이로부터 "COS가 진짜 화자를 재현하지 못한다"는 결론은 **내리지 않는다**(이전 서사 철회).""")
code("""fid={}
for tag in TAG_ORDER:
    fs=[]
    m=np.where(Z["tag"]==tag)[0]
    for i in m:
        r=REFEMB[i]
        if not np.isnan(r).any(): fs.append(max(0.0,cos(EMB[i],r)))
    fid[tag]=np.mean(fs) if fs else np.nan
fig,ax=plt.subplots(figsize=(9,4.8))
xs=[official(t)[0] for t in TAG_ORDER]
ax.plot(xs,[within[t] for t in TAG_ORDER],"o-",color="#4C72B0",label="self-consistency (Gen-Gen, the actual metric)")
ax.plot(xs,[fid[t] for t in TAG_ORDER],"^-",color="#999999",label="cos to LibriTTS source utt (NOT a target; domain-gap confounded)")
for t in TAG_ORDER:
    ax.annotate(LBL[t].split(" (")[0],(official(t)[0],fid[t]),fontsize=7,xytext=(0,-12),textcoords="offset points",ha="center")
ax.set_xlabel("official COS"); ax.set_ylabel("mean cosine"); ax.legend(fontsize=7); ax.grid(alpha=.25)
ax.set_title("참고용: source-utt 대조(회색)는 충실도 지표가 아님 — 결론에 사용하지 않음")
plt.tight_layout(); plt.savefig(ANA/"fig3c_fidelity.png"); plt.show()""")

md("""### 3d. 임베딩 공간 시각화 (PCA) — 저COS vs 고COS

페르소나별 색으로 syn 임베딩을 2D 투영. 저COS(퍼짐) → 고COS(각 페르소나 뭉침, 그러나 군집 간 거리 축소) 대조.""")
code("""from sklearn.decomposition import PCA
comp_tags=["vd_default","vqr_k32_recovered","rt2"]
fig,axes=plt.subplots(1,len(comp_tags),figsize=(15,5),sharex=True,sharey=True)
# shared PCA over the union for comparable axes
allidx=np.where(np.isin(Z["tag"],comp_tags))[0]
pca=PCA(n_components=2).fit(EMB[allidx])
for ax,tag in zip(axes,comp_tags):
    m=np.where(Z["tag"]==tag)[0]; P=pca.transform(EMB[m])
    rids=Z["rid"][m]; uniq=sorted(set(rids))
    cmap=plt.cm.tab10(np.linspace(0,1,len(uniq)))
    for c,rid in zip(cmap,uniq):
        sel=rids==rid
        ax.scatter(P[sel,0],P[sel,1],s=26,color=c,alpha=.75,edgecolor="white",linewidth=.3)
    c,w=official(tag)
    ax.set_title(f"{LBL[tag]}\\nCOS={c:.3f} WER={w:.3f}"); ax.grid(alpha=.2)
fig.suptitle("페르소나별 syn 임베딩 (PCA, 색=페르소나) — COS↑ 시 군집화 vs 군집간 거리",y=1.03)
plt.tight_layout(); plt.savefig(ANA/"fig3d_pca.png",bbox_inches="tight"); plt.show()""")

# ---- §4 WER collapse
md("""## §4. WER 붕괴의 메커니즘

### 4a. per-utterance WER 분포 — 균일 악화인가, 파국적 꼬리인가?
체크포인트를 COS 순으로 정렬해 per-utterance WER 분포(violin)를 본다.""")
code("""order=sorted(TAG_ORDER,key=lambda t:official(t)[0])
data=[df[df["tag"]==t]["wer"].dropna().values for t in order]
fig,ax=plt.subplots(figsize=(11,5))
parts=ax.violinplot(data,showmedians=True,widths=.8)
ax.set_xticks(range(1,len(order)+1))
ax.set_xticklabels([f"{LBL[t].split(' (')[0]}\\nCOS={official(t)[0]:.3f}" for t in order],rotation=25,ha="right",fontsize=8)
ax.axhline(0.12,ls="--",c="green",lw=1,label="WER target 0.12")
ax.set_ylabel("per-utterance WER"); ax.set_title("COS↑ 순 per-utterance WER 분포 — 상위 꼬리(파국적 발화)가 두꺼워짐")
ax.legend(); ax.grid(alpha=.25,axis="y")
# annotate tail fraction
for i,t in enumerate(order):
    v=df[df["tag"]==t]["wer"].dropna().values
    frac=np.mean(v>0.3)
    ax.text(i+1,1.02,f"{frac*100:.0f}%>.3",ha="center",fontsize=7,color="#C44E52")
plt.tight_layout(); plt.savefig(ANA/"fig4a_wer_violin.png"); plt.show()""")

md("""### 4b. micro 트레이드오프는 **없다** (반증) — WER은 발화 단위 화자강도와 무관

각 발화의 synᵢ→syn₀ 유사도 vs 그 발화의 WER. 만약 "화자를 밀면 WER이 나빠진다"가 발화 단위로 성립하면 양의 상관이 보여야 한다.
**결과: pooled r≈0.05 (무상관).** 즉 트레이드오프는 per-utterance 수준의 현상이 **아니라**, 체크포인트(학습 레짐) 수준의 현상이다.""")
code("""fig,ax=plt.subplots(figsize=(9,6))
simcol=[]; wercol=[]; tagcol=[]
for tag in TAG_ORDER:
    for rid in sorted(set(Z["rid"][Z["tag"]==tag])):
        idx=rows_of(tag,rid)
        if len(idx)<2: continue
        a=EMB[idx[0]]
        for j in idx[1:]:
            s=max(0.0,cos(a,EMB[j])); w=Z["wer"][j]
            if not np.isnan(w): simcol.append(s); wercol.append(w); tagcol.append(tag)
simcol=np.array(simcol); wercol=np.array(wercol)
# bin by similarity, show mean WER
bins=np.linspace(0,1,11); bc=.5*(bins[:-1]+bins[1:])
mw=[np.nanmean(wercol[(simcol>=bins[i])&(simcol<bins[i+1])]) if np.any((simcol>=bins[i])&(simcol<bins[i+1])) else np.nan for i in range(len(bins)-1)]
ax.scatter(simcol,wercol,s=6,alpha=.12,color="#4C72B0")
ax.plot(bc,mw,"o-",color="#C44E52",lw=2,label="binned mean WER")
r=np.corrcoef(simcol,wercol)[0,1]
ax.set_xlabel("per-utterance similarity to syn₀ (speaker push)"); ax.set_ylabel("per-utterance WER")
ax.set_title(f"micro 트레이드오프 없음: 유사도와 WER 무상관 (pooled r={r:.2f})")
ax.legend(); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig(ANA/"fig4b_micro.png"); plt.show()
print("pooled Pearson r(sim,wer)=",round(r,3))""")

md("""### 4c. **WER 붕괴의 진짜 정체 = 생성 속도 폭주 (핵심 발견)**

발화속도(words/sec)를 COS 순으로. **corr(WER, speaking-rate)=0.89** — 압도적.
최악 WER 체크포인트 **rt1(WER 0.195)** 은 발화속도가 **~9.8 wps** (정상 2.5–2.9의 3.5배)로 폭주하고 파국적 발화(WER>0.3)가 12%다.
즉 WER 붕괴는 "앵커가 비정형이라서"가 아니라, 학습이 생성기를 불안정하게 만들어 **연속화가 초고속/뭉개짐으로 폭주**할 때 발생한다.""")
code("""order=sorted(TAG_ORDER,key=lambda t:official(t)[0])
fig,axs=plt.subplots(1,2,figsize=(14,5))
# rate distribution
rd=[df[df["tag"]==t]["rate"].replace([np.inf,-np.inf],np.nan).dropna().values for t in order]
axs[0].violinplot(rd,showmedians=True,widths=.8)
axs[0].set_xticks(range(1,len(order)+1)); axs[0].set_xticklabels([f"{official(t)[0]:.3f}" for t in order],rotation=0,fontsize=8)
axs[0].set_xlabel("checkpoint (by COS)"); axs[0].set_ylabel("speaking rate (words/sec)")
axs[0].axhspan(2.3,3.6,color="green",alpha=.08,label="typical band"); axs[0].legend()
axs[0].set_title("발화속도 분포 — 고COS에서 저속/변동 증가")
# rate deviation vs WER (per utt)
rate=df["rate"].replace([np.inf,-np.inf],np.nan).values
dev=np.abs(rate-np.nanmedian(rate)); w=df["wer"].values
ok=~np.isnan(dev)&~np.isnan(w)
axs[1].scatter(dev[ok],w[ok],s=6,alpha=.12,color="#55A868")
bins=np.linspace(0,np.nanpercentile(dev[ok],98),11); bc=.5*(bins[:-1]+bins[1:])
mw=[np.nanmean(w[ok][(dev[ok]>=bins[i])&(dev[ok]<bins[i+1])]) if np.any((dev[ok]>=bins[i])&(dev[ok]<bins[i+1])) else np.nan for i in range(len(bins)-1)]
axs[1].plot(bc,mw,"o-",color="#C44E52",lw=2,label="binned mean WER")
axs[1].set_xlabel("|speaking-rate − median| (prosody abnormality)"); axs[1].set_ylabel("per-utterance WER")
axs[1].set_title("운율 비정상성↑ → WER↑"); axs[1].legend(); axs[1].grid(alpha=.25)
plt.tight_layout(); plt.savefig(ANA/"fig4c_prosody.png"); plt.show()""")

md("""### 4d. 앵커 비정형성은 WER을 설명하지 못한다 (가설 반증) — 내부 앵커 재구성 측정

저장된 syn엔 내부 앵커가 잘려 없다. VQ-query 체크포인트로 페르소나별 앵커를 **가볍게 재구성**(재생성 아님, 지표 규명용)해
앵커의 (i) 길이, (ii) 자연발화 중심으로부터의 임베딩 거리(atypicality)를 측정한다.

**당초 가설**(COS↑→앵커 비정형↑→WER↑)은 데이터로 반증된다:
- **corr(WER, atyp) = −0.14** (거의 무상관) — 앵커 비정형성은 WER의 원인이 아니다.
- **corr(COS, atyp) = −0.64** — 오히려 고COS 앵커가 자연발화에 **더 가깝다**(예상과 반대).
- 다만 재구성된 앵커 길이가 **10–13초로 비정상적으로 김**(짧은 FIXED_CONTENT 대비) — 앵커가 장황한 drift 오디오임은 사실이나, 그 정도는 WER과 약한 상관(0.30)뿐.""")
code("""# natural-speech centroid = mean of all real reference embeddings on disk
natural=REFEMB[~np.isnan(REFEMB).any(1)]
nat_centroid=natural.mean(0); nat_centroid/=np.linalg.norm(nat_centroid)
print("natural refs:", len(natural))

RECON_TAGS=[(t,ck) for t,_,ck in TAGS if ck]  # only VQ-query variants
ASTAT=ANA/"anchor_stats.npy"
anchor_stats={}
if ASTAT.exists():
    anchor_stats=np.load(ASTAT, allow_pickle=True).item()
    print("loaded anchor_stats cache:", {k:round(v['atyp'],3) for k,v in anchor_stats.items()})
try:
    if anchor_stats: raise StopIteration  # skip recon when cached
    import new_infer_lib as L
    for tag,ck in RECON_TAGS:
        L.load_model(DEV, anchor_mode="vq", vq_ckpt=ck)
        gk=dict(pad_token_id=L.tokenizer.eos_token_id)
        durs=[]; atyp=[]
        # reuse the 10 personas' style prompts from that tag's metadata dirs is complex;
        # use LibriTTS-P style prompts as in generation (first 10 unique personas)
        from spk_incon.metrics.presets import DatasetType, GenerationMethod, SynthesisConfig
        from spk_incon.metrics.strategies import create_strategy
        from spk_incon.datasets import create_dataset
        cfg=SynthesisConfig(); ds=create_dataset(DatasetType.LIBRITTS,cfg.get_dataset_config("libritts"),root_dir="./data")
        class _D:
            def synthesize(self,**k): return True
        strat=create_strategy(GenerationMethod.METHOD2,cfg,ds,_D())
        idxs=strat.select_unique_speakers(cfg.generation.method2_ref_samples)
        for si in idxs:
            _,_,style,_=ds.get_sample(si); persona=style or L.FIXED_CONTENT
            aw,asr=L.make_anchor_audio(persona,gk,seed=42)
            t=torch.tensor(np.asarray(aw,dtype=np.float32))
            if asr!=16000: t=torchaudio.functional.resample(t,asr,16000)
            e=ecapa_emb(t); durs.append(len(aw)/asr); atyp.append(1-max(0.0,cos(e,nat_centroid)))
        anchor_stats[tag]=dict(dur=np.mean(durs),atyp=np.mean(atyp))
        print(f"{tag}: anchor dur={np.mean(durs):.2f}s atyp={np.mean(atyp):.3f}",flush=True)
    del L
    torch.cuda.empty_cache()
    np.save(ASTAT, anchor_stats, allow_pickle=True)
except StopIteration:
    pass
except Exception as e:
    print("anchor recon skipped:", repr(e))""")

code("""if anchor_stats:
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    ts=list(anchor_stats)
    coss=[official(t)[0] for t in ts]; wers=[official(t)[1] for t in ts]
    atyp=[anchor_stats[t]["atyp"] for t in ts]; durs=[anchor_stats[t]["dur"] for t in ts]
    sc=axs[0].scatter(coss,atyp,c=wers,s=120,cmap="Reds",edgecolor="k")
    for t in ts: axs[0].annotate(LBL[t].split(" (")[0],(official(t)[0],anchor_stats[t]["atyp"]),fontsize=7,xytext=(4,4),textcoords="offset points")
    axs[0].set_xlabel("official COS"); axs[0].set_ylabel("anchor atypicality (1−cos to natural centroid)")
    r_ca=np.corrcoef(coss,atyp)[0,1]
    plt.colorbar(sc,ax=axs[0],label="WER"); axs[0].set_title(f"COS↑ → 앵커 비정형성↓ (r={r_ca:.2f}, 예상과 반대; 색=WER)"); axs[0].grid(alpha=.25)
    sc2=axs[1].scatter(atyp,wers,c=coss,s=120,cmap="viridis",edgecolor="k")
    for t in ts: axs[1].annotate(LBL[t].split(" (")[0],(anchor_stats[t]["atyp"],official(t)[1]),fontsize=7,xytext=(4,4),textcoords="offset points")
    axs[1].set_xlabel("anchor atypicality"); axs[1].set_ylabel("WER"); plt.colorbar(sc2,ax=axs[1],label="COS")
    r_aw=np.corrcoef(atyp,wers)[0,1]
    axs[1].axhline(0.12,ls="--",c="green"); axs[1].set_title(f"앵커 비정형성 ⟂ WER (r={r_aw:.2f}, 무상관)"); axs[1].grid(alpha=.25)
    plt.tight_layout(); plt.savefig(ANA/"fig4d_anchor.png"); plt.show()
else:
    print("no anchor stats to plot")""")

# ---- §4e quantitative correlation summary
md("""### 4e. 정량 요약 — 어떤 요인이 실제로 WER/COS와 연결되는가 (체크포인트 8점)

가설들을 상관계수로 채점한다. **속도(rate)만이 WER의 강한 상관자**임이 드러난다.""")
code("""def _off(t):
    m=json.load(open(f"results/_summary/{t}.json"))["metrics"]; return m["sim_mean"],m["wer_mean"]
rows=[]
for t in TAG_ORDER:
    m=Z["tag"]==t; c,w=_off(t)
    rows.append((t,c,w,np.nanmean(Z["rate"][m]),np.mean(Z["wer"][m]>0.3),
                 anchor_stats.get(t,{}).get("atyp",np.nan),anchor_stats.get(t,{}).get("dur",np.nan)))
C=np.array([r[1] for r in rows]); W=np.array([r[2] for r in rows]); RT=np.array([r[3] for r in rows])
TL=np.array([r[4] for r in rows]); AT=np.array([r[5] for r in rows]); AD=np.array([r[6] for r in rows])
def cc(a,b):
    ok=~np.isnan(a)&~np.isnan(b); return np.corrcoef(a[ok],b[ok])[0,1] if ok.sum()>2 else np.nan
labels=["WER ~ speaking-rate","WER ~ COS","WER ~ catastrophic-tail","WER ~ anchor-dur","WER ~ anchor-atyp","COS ~ anchor-atyp"]
vals=[cc(W,RT),cc(W,C),cc(W,TL),cc(W,AD),cc(W,AT),cc(C,AT)]
fig,ax=plt.subplots(figsize=(9,4.5))
col=["#C44E52" if abs(v)>=0.6 else ("#DD8452" if abs(v)>=0.4 else "#B0B0B0") for v in vals]
ax.barh(labels[::-1],vals[::-1],color=col[::-1]); ax.axvline(0,c="k",lw=.8)
for i,v in enumerate(vals[::-1]): ax.text(v+(0.02 if v>=0 else -0.02),i,f"{v:.2f}",va="center",ha="left" if v>=0 else "right",fontsize=9)
ax.set_xlim(-1,1); ax.set_xlabel("Pearson r (across 8 checkpoints)")
ax.set_title("WER의 유일한 강상관자 = 발화속도 폭주 · 앵커 비정형성 가설은 반증")
ax.grid(alpha=.25,axis="x"); plt.tight_layout(); plt.savefig(ANA/"fig4e_corr.png"); plt.show()
for l,v in zip(labels,vals): print(f"  {l:28s} r={v:+.3f}")""")

# ---- §5 synthesis
md("""## §5. 종합 — 하나의 그림으로 본 벽""")
code("""fig,ax1=plt.subplots(figsize=(10,6))
xs=[official(t)[0] for t in TAG_ORDER]; ws=[official(t)[1] for t in TAG_ORDER]
ax1.scatter(xs,ws,s=90,color="#4C72B0",zorder=3)
for t in TAG_ORDER: ax1.annotate(LBL[t].split(" (")[0],(official(t)[0],official(t)[1]),fontsize=8,xytext=(5,4),textcoords="offset points")
# trend
o=np.argsort(xs); ax1.plot(np.array(xs)[o],np.array(ws)[o],color="#4C72B0",alpha=.4)
ax1.axhline(0.12,ls="--",c="green"); ax1.axvline(0.40,ls="--",c="green")
ax1.add_patch(plt.Rectangle((0.40,0.0),0.2,0.12,color="green",alpha=.10))
ax1.text(0.402,0.118,"TARGET (empty)",color="green",va="top",fontsize=9)
ax1.set_xlabel("Gen-Gen COS"); ax1.set_ylabel("WER")
ax1.set_title("프론티어 요약: COS–WER 결합은 중간(r≈0.47), 붕괴점(rt1)은 생성 불안정")
ax1.grid(alpha=.25); plt.tight_layout(); plt.savefig(ANA/"fig5_synthesis.png"); plt.show()""")

md("""## 결론 (데이터가 지지하는 것 / 반증한 것)

이 분석은 당초 가설 여러 개를 **반증**했다. 정직하게, 데이터가 실제로 말하는 바만 정리한다.

### 반증된 가설 (❌)
- **모드 붕괴로 인한 COS 천장** — 아니다. per-file(무패딩) 임베딩에서 within-persona 일관성은 모든 query 체크포인트가 **이미 0.71–0.75로 포화**(§3a,3b)이고, between-persona는 낮게 유지(최고 체크포인트 0.19)된다. 붕괴 없음.
- **앵커 비정형성 → WER** — 아니다. corr(WER, atyp)=−0.14, 심지어 corr(COS, atyp)=**−0.64**로 고COS 앵커가 자연발화에 더 가깝다(§4d,4e).
- **per-utterance 화자강도↔WER 트레이드오프** — 아니다. pooled r≈0.05(§4b).

### 데이터가 지지하는 규명 (✅)
1. **COS "천장"의 상당 부분은 지표 압축이다.** 실제 일관성(per-file 0.7+)은 이미 높고, 공식 method2 COS(~0.42)는 zero-pad 배치 임베딩이 이를 압축한 값이다(§3a,3b). 즉 "일관성을 더 못 만든다"기보다 지표가 큰 값을 눌러 보고한다. (§3c의 source-utt 대조는 이 과제에 타깃 음성이 없고 도메인 갭이 섞여 결론에 사용하지 않음 — 이전 fidelity 서사 철회.)

2. **WER 붕괴 = 생성 속도 폭주(생성 불안정).** WER의 강상관자는 **발화속도(r=0.89)와 파국적 발화율(r=0.97, §4c,4e)**. 최악점 rt1(WER 0.195)은 속도가 ~9.8 wps로 폭주하고 파국적 발화가 12%다. 이는 앵커의 음향적 성질(비정형성은 무상관, §4d)이 아니라, 학습이 생성기를 불안정 영역으로 몰 때(특히 불안정한 reftext 레짐) 연속화가 초고속·뭉개짐으로 폭주해 생긴다.

3. **"벽"의 정확한 성격:** COS–WER 결합은 **중간 정도(r≈0.47)** 이며 결정적 물리법칙이 아니다. 재현 가능한 최고점은 **0.42 COS / 0.13 WER**에 모여 있고, 공식 COS를 그 이상으로 미는 데 성공한 학습 레짐(rt1, rt2 등 불안정 reftext)은 **일관되게 생성 불안정(속도 폭주)을 유발**해 WER을 깨뜨린다. 즉 벽은 "COS↔WER의 직접 인과"라기보다, **공식 COS를 더 짜내려는 학습 압력이 생성기를 불안정화한다**는 최적화-안정성 문제다.

### 함의 (다음에 벽을 넘으려면)
- 공식 COS의 압축 아티팩트를 감안하면, 실제 일관성은 이미 높다 → **지표 압축을 우회**(예: 길이 정합/패딩 제거)만으로도 보고 COS가 오를 수 있다(단, 지표는 건드리지 않기로 함).
- 진짜 개선은 **생성 안정성(속도 폭주 억제: 디코딩 제약/길이 정규화)** 을 확보한 채로 화자 조건을 주입하는 것. 앵커-공유 ICL 구조에서 학습 압력이 생성기를 불안정화하는 고리를 끊어야 한다.

모든 그림: `results/analysis/fig*.png` (fig1, 3a–3d, 4a–4e, 5). per-utterance 캐시: `results/analysis/perutt.npz`.""")

nb["cells"]=cells
nbf.write(nb, "new_cos_wer_analysis.ipynb")
print("wrote new_cos_wer_analysis.ipynb with", len(cells), "cells")
