# Models & data for L2 phone judging — post-M1 research report

Status: FINAL · Date: 2026-08-20 · Scope: feeds the re-attempt of the segmental
judge line after the M1 negative.

> **TL;DR (plain language).**
> The model question is settled: use **XLS-R-300M frozen + a tiny trainable
> head** (Apache-2.0, clean to ship; Charsiu is literally built on it, so we
> keep Charsiu's phone-recognition power without its license hole). The data
> question has bad news and good news. Bad: "huge open non-native English
> corpora" do not exist — error-labeled /ð/ /θ/ audio is measured in *minutes*,
> not hours, and Russian-accented English is only available through a gated
> commercial set. Good: we don't need "huge" — correct-speech mass is free
> (Common Voice, CC0), our own corpora already hold 2,370 error tokens across
> both dental fricatives, and synthetic accented audio is feasible with free
> permissive TTS. So: **training the judge — realistic. Training on huge
> datasets — not, and unnecessary. The "huge" ambition gets rebuilt as CC0 mass
> + small real error data + synthetic accents.** Details and every number below.

## 1. Why this report exists

M1's /ð/-judge line closed as a documented negative: the honest,
speaker-disjoint ceiling was ≈ 0.69 pooled AUC (L2-ARCTIC 0.596) against a
shipping bar of precision ≥ 0.90 at recall ≥ 0.40. Alignment was never the
problem (0.90 mean confidence) — **discrimination** was. This report answers
three questions before any further training money is spent:

1. Which open models can OpenSchwa use — and realistically train on one
   RTX 4060 laptop — for phone-level L2 error judging (/ð/ vs z,d,v;
   /θ/ vs s,t,f)?
2. Which non-native English corpora exist with **connected speech** (read
   sentences, read paragraphs, spontaneous), and which combination reaches a
   trainable scale?
3. Is "train it on huge datasets" a realistic demand at all? — answered with
   numbers in §5.

Constraints that shaped the research: closed-set contrast (the engine always
knows the target phone sequence — ASR robustness is the enemy, not the goal);
CPU-friendly local inference with an ONNX escape hatch; AGPL project that
must be able to explain where every model weight came from; training on a
laptop GPU (8 GB VRAM), dev on Apple silicon; corpora are never committed,
only used offline.

## 2. Method

Two independent research passes (models / corpora) over primary sources —
HF model cards, GitHub LICENSE files (raw), ELRA/LDC catalogues, corpus
sites, papers — followed by this synthesis. Corpus label counts in §4.0 were
measured locally with the repo's own dataset adapters. Anything unverifiable
is marked **UNVERIFIED** — never guessed. Gated or paywalled items are listed
with their gate, not worked around.

## 3. Models

### 3.1 The pick: XLS-R-300M, frozen + a small head

**facebook/wav2vec2-xls-r-300m** — Apache-2.0, 300M params, multilingual
self-supervised encoder with strong phone-discrimination features (ABX / MLS
cross-lingual phoneme results). It is the exact backbone Charsiu is built on
(Charsiu = XLS-R-300M + a CTC phone head), so it reproduces Charsiu's
phone-level signal **without the Charsiu checkpoint's missing license** (§3.3).
For our data sizes (93–1,676 positives per class, §4.0), fine-tuning a big
model is the wrong tool: freeze the encoder, train a small head.

Concrete recipe (from the research pass, adapted to our CLIs):

1. Frameworks: HF transformers (Wav2Vec2Model) + PyTorch; export via
   torch.onnx / optimum → ONNX Runtime for CPU inference.
2. Input: 16 kHz; run the whole word/utterance through the frozen encoder,
   then slice the target phone's frames using the alignment we already trust
   (Charsiu CTC or MFA). Segment-level, not full-utterance.
3. Features: mean + std pool of hidden states from **layers 12–20** (middle
   layers encode phone identity best) → 1024-d vector per phone segment.
4. Head: 2-layer MLP (1024 → 256 → N). Per contrast N classes: /ð/ =
   {ð,z,d,v}, /θ/ = {θ,s,t,f}. Consider one binary head per substitution
   ("correct vs /z/") — it targets the rare class directly instead of
   lumping 93 vs 1676 examples into one softmax.
5. Class imbalance: weighted cross-entropy or focal loss + oversampling of
   the rare /ð/→/z/ class; evaluate AUROC + P@R=0.40; speaker-disjoint splits
   (hold out *speakers*, never shuffled frames).
6. Optimizer: AdamW, lr 1e-3 (head) / 1e-4 (LoRA), cosine decay, 20–50
   epochs, early-stop on val AUC.
7. VRAM: head-only ≈ 1.5–2.5 GB (cache encoder features to disk once — no
   GPU needed to re-run the frozen encoder). LoRA r=8–16 on query/key/value +
   output projection with gradient checkpointing ≈ 6–7 GB on the 4060.
8. Augmentation: time-masking + mild noise only. **Avoid aggressive frequency
   masking** — the /θ/ vs /s/ and /ð/ vs /z/ cues live in high-frequency
   spectral shape, which freq-masking destroys.

**Is a frozen-encoder + linear head worth trying first? Yes — do it first.**
A linear probe (logistic regression / linear SVM on pooled frozen features)
is the correct experiment before any LoRA: 93 examples cannot support
fine-tuning 300M–1B weights; SSL features are already phone-discriminative
(TACL 2024 probing study; "MDD Without Model Training", arXiv 2511.20107,
2025 — frozen features + shallow classifier already work for MDD); and it
isolates *where* the 0.69 AUC came from — if a frozen probe beats it, M1's
CTC head was the bottleneck; if not, we need deeper heads or layer selection.

### 3.2 Model decision table

Legend: **use** = license-clean, ship-ready · **borrow** = usable with a
caveat · **test** = worth one experiment · **skip** = measured reason.
"NC" = non-commercial clause (incompatible with shipping AGPL downstream).

| model | family | params | license (primary source) | phone signal | 8 GB fine-tune | CPU+ONNX | verdict |
|---|---|---|---|---|---|---|---|
| XLS-R-300M | SSL | 300M | Apache-2.0 — [HF](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | frame features, strong cross-lingual phones | frozen head / LoRA | good | **use (first)** |
| XLS-R-1B | SSL | 1B | Apache-2.0 — [HF](https://huggingface.co/facebook/wav2vec2-xls-r-1b) | better features | frozen head only; LoRA tight | OK, slower | test |
| XLS-R-2B | SSL | 2B | Apache-2.0 — [HF](https://huggingface.co/facebook/wav2vec2-xls-r-2b) | better features | frozen head only | heavy on CPU | skip (CPU target) |
| wav2vec2-base | SSL | 95M | Apache-2.0 — [HF](https://huggingface.co/facebook/wav2vec2-base) | frame features | head-only / full | good | borrow |
| HuBERT base/large | SSL | 95M/316M | Apache-2.0 — [HF](https://huggingface.co/facebook/hubert-base-ls960) | frame features | head / LoRA | good | test |
| WavLM base/large | SSL | 95M/316M | code MIT — [unilm](https://github.com/microsoft/unilm/blob/master/LICENSE); HF card has no license tag → **weights UNVERIFIED** | frame features, denoise-robust | head / LoRA | good | borrow (license caveat) |
| WavLabLM | SSL | ~0.3B (UNVERIFIED) | weights CC-BY-4.0 — [HF](https://huggingface.co/espnet/WavLabLM-MS-40k); code MIT | universal SSL features | head / LoRA | good | test |
| MMS 300M/1B | SSL/ASR | 300M/1B | **CC-BY-NC-4.0** — [fairseq MMS README](https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md) | ASR features, not phone-native | LoRA / adapter | OK | skip (NC) |
| Charsiu 300M | phone rec. | 300M | code MIT — [GitHub](https://github.com/lingjzhu/charsiu/blob/main/LICENSE); **HF checkpoint has no license file → weights UNVERIFIED** | explicit IPA CTC phone posteriors | as XLS-R | good | borrow (alignment/research); don't ship weights |
| XLS-R CTC phone heads (espeak/CV) | phone rec. | 300M | Apache-2.0 (XLS-R base) | explicit CTC phone posteriors | head / LoRA | good | borrow |
| Allosaurus | phone rec. | ~30M | GPL-3.0 — [GitHub](https://github.com/xinjli/allosaurus/blob/master/LICENSE) | universal phone posteriors | small | good | skip (outdated) |
| CharsiuG2P | G2P | ~0.1–0.3B (UNVERIFIED) | code MIT — [GitHub](https://github.com/lingjzhu/charsiug2p/blob/main/LICENSE); weights UNVERIFIED | none (text→IPA) | n/a | n/a | skip as judge; borrow for lexicon |
| Whisper tiny/base/small | ASR | 39M/74M/244M | MIT — [OpenAI](https://github.com/openai/whisper/blob/main/LICENSE) (HF card says apache-2.0; both permissive) | encoder features only, no phones; accent-robust (enemy) | tiny/base full FT; small LoRA | good | test (encoder only) |
| Distil-Whisper | ASR | 166M/756M | MIT — [HF](https://huggingface.co/distil-whisper/distil-large-v3) | encoder features; EN-only | LoRA | good | skip |
| OWSM v3.1 | ASR | ~1B | CC-BY-4.0 — [HF](https://huggingface.co/espnet/owsm_v3.1_ebf) | E-Branchformer encoder; subword | LoRA only | encoder heavy | test |
| NVIDIA Parakeet 0.6B | ASR | 600M | CC-BY-4.0 — [HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | FastConformer encoder; subword | LoRA only | OK | test |
| NVIDIA Canary 1B | ASR | 1B | **CC-BY-NC-4.0** — [HF](https://huggingface.co/nvidia/canary-1b) | FastConformer encoder | LoRA | heavy | skip (NC) |
| Qwen3-ASR 0.6B/1.7B | ASR | 0.6B/1.7B | Apache-2.0 — [HF](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | encoder features; 52 langs | LoRA | 0.6B OK | test |
| Meta Omnilingual ASR 300M/1B | ASR/SSL | 300M/1B | Apache-2.0 — [Meta blog](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition); data CC-BY | encoder features; 1600 langs | frozen / LoRA | 300M OK | test |
| Piper | TTS | voices ~20–30 MB ONNX | engine MIT (old rhasspy/piper); piper1 GPL-3.0 — [repo](https://github.com/OHF-Voice/piper1-gpl); **voices per-voice licensed** | n/a (synthesis) | n/a | real-time CPU | use (data gen, check voice license) |
| StyleTTS2 | TTS | ~160M | MIT — [GitHub](https://github.com/yl4579/StyleTTS2/blob/main/LICENSE) | n/a | n/a | OK | use (data gen) |
| CosyVoice / CosyVoice2 | TTS/VC | 300M/0.5B | Apache-2.0 — [GitHub](https://github.com/FunAudioLLM/CosyVoice/blob/main/LICENSE) | n/a | n/a | slow CPU | use (cloning + accented data) |
| F5-TTS | TTS | ~335M | code MIT — [GitHub](https://github.com/SWivid/F5-TTS/blob/main/LICENSE); **weights CC-BY-NC** | n/a | n/a | OK | borrow (code); avoid weights |
| MeloTTS | TTS | ~ (small) | MIT — [HF](https://huggingface.co/myshell-ai/MeloTTS-English) | n/a | n/a | OK | use (data gen) |
| XTTS-v2 | TTS/VC | ~1B | **weights CPML (non-commercial)**; code MPL-2.0 | n/a | n/a | OK | skip (CPML) |
| OpenVoice | VC | ~ (small) | MIT — [GitHub](https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE) | n/a | n/a | OK | use (cloning) |

### 3.3 The Charsiu license, settled

GitHub code (lingjzhu/charsiu) is **MIT** (verified from the raw LICENSE, both
branches). The HF checkpoint has **no license file** — which matches the
registry's "unknown (charsiu)" — so weight-redistribution terms are
UNVERIFIED: an MIT code repo does not automatically license the checkpoint.
Practical consequence: keep Charsiu for alignment/research, but the
ship-ready path is **XLS-R-300M directly (Apache-2.0) + our own phone head**.
CharsiuG2P code is also MIT.

### 3.4 2025–2026 additions worth knowing

Meta **Omnilingual ASR** (300M/1B, 1600 langs, Apache-2.0, Nov 2025) and
**Qwen3-ASR** (Apache-2.0, 2026) as encoder donors; **WavLabLM** (CC-BY-4.0
weights); refreshed **Parakeet v3**, **OWSM v3.2**, **Whisper-large-v3-turbo**.
Skip: SenseVoice, Spirit-LM, SeamlessM4T v2 (all NC).

### 3.5 MDD evidence (papers to anchor the re-attempt)

- L2-ARCTIC as the MDD benchmark: Zhao et al. 2018, [ISCA](https://www.isca-archive.org/interspeech_2018/zhao18b_interspeech.pdf).
- SSL-for-MDD: Yan et al. 2021 "Explore wav2vec 2.0 for MDD"; [arXiv 2204.03863](https://arxiv.org/abs/2204.03863) (SpeechOcean762); [arXiv 2311.07037](https://arxiv.org/abs/2311.07037) (phonological-level MDD); momentum pseudo-labeling (Interspeech 2022); "Automatic Pronunciation Assessment — A Review" (ACL Findings 2023).
- Frozen-feature MDD (supports §3.1): [arXiv 2511.20107](https://arxiv.org/abs/2511.20107) (2025, training-free) and [arXiv 2506.01156](https://arxiv.org/abs/2506.01156) (2025, low-resource).
- SSL probing: TACL 2024 [2024.tacl-1.21](https://aclanthology.org/2024.tacl-1.21).
- Dental fricatives in L2: most published work is acoustic/descriptive, not
  classifier-based — our 93-example /ð/→/z/ setting is genuinely
  under-studied, which argues for frozen-feature methods over big fine-tunes.

## 4. Datasets

### 4.0 What our own corpora already contain (counted 2026-08-20)

Counted with the repo's own adapters on the local copies — these supersede
earlier round numbers:

| corpus | target | utterances | tokens | error tokens | error rate | dominant realizations |
|---|---|---|---|---|---|---|
| L2-ARCTIC | θ | 778 | 894 | 397 | 44% | t 214 · s 106 · d 29 · f 11 |
| L2-ARCTIC | ð | 2454 | 3651 | 1973 | 54% | d 1676 · z 93 · l 52 · t 41 |
| speechocean762 | θ | 602 | 649 | 6 | <1% | — (error-sparse) |
| speechocean762 | ð | 2414 | 3027 | 6 | <1% | — (error-sparse) |

What this means:

- **Both dental fricatives are attackable with real labels today.** /θ/
  errors exist in all six L1 groups (arabic 26, hindi 106, mandarin 83,
  spanish 85, korean 29, vietnamese 68). Label starvation overall is not the
  problem M1 had.
- **But the headline contrast is the thinnest one.** The classic /ð/→/z/
  ("zis") has only **93 positive examples** in L2-ARCTIC; /θ/→/s/ has 106.
  The corpora's dominant realizations are *stops* (d: 1676, t: 214). Any
  re-attempt must either keep the full confusion set {z,d,v} / {s,t,f}
  (M1's design — the mass lives in the stops) or acquire more sibilant
  realization mass (Speech Accent Archive, synthetic injection).
- **speechocean762 is correct-class mass + calibration only** — 12 error
  tokens total across both phones. It cannot train error discrimination; it
  can anchor "correct /ð/ sounds like this".

### 4.1 Corpus table (non-native English, connected speech)

| corpus | hours | speakers | L1s | style | annotation | license (primary) | train-weights-OK | access |
|---|---|---|---|---|---|---|---|---|
| L2-ARCTIC (held) | ~27 h (UNVERIFIED) | 24 | 6 (ar/hi/ko/zh/es/vi) | read sentences | **phone errors** (sub/del/ins TextGrids) | CC BY-NC 4.0 — [TAMU](https://psi.engr.tamu.edu/l2-arctic-corpus) | no (NC) | free |
| SpeechOcean762 (held) | ~9–10 h (UNVERIFIED) | 250 | 1 (zh) | read sentences | 5-expert phone scores | CC BY 4.0 — [GitHub](https://github.com/jimbozhang/speechocean762) | yes | free |
| Speech Accent Archive | ~3 h | ~2,959 | 214+ L1s (incl. ru) | **one read paragraph/spk** | orthographic + broad phonetic | official terms unclear (UNVERIFIED); Kaggle mirror CC BY-NC-SA — [Kaggle](https://www.kaggle.com/datasets/rtatman/speech-accent-archive) | no (mirror) | site browse / Kaggle |
| ISLE (ELRA-S0083) | 17 h 54 m | 46 | 2 (de, it) | read sentences | **phone errors** | Non-commercial (ELRA) — [catalogue](https://catalogue.elra.info/en-us/repository/browse/ELRA-S0083) | no (NC) | free w/ registration |
| EpaDB | ~1 h (UNVERIFIED) | 50 | 1 (es-AR) | read short utt. | phoneme annotations | UNVERIFIED — [GitHub](https://github.com/JazminVidal/gop-dnn-epadb), [HF](https://huggingface.co/datasets/KoelLabs/EpaDB) | UNVERIFIED | email author |
| AESRC2020 | 160 h (≈20 h × 8) | ~480 | 8 incl. **Russia** | read speech | transcripts only | Datatang commercial; not openly downloadable — [repo](https://github.com/R1ckShi/AESRC2020), [paper](https://arxiv.org/abs/2102.10233) | no | gated (Datatang) |
| Common Voice EN | ~3,000+ h validated | ~88,900+ | accent = native EN region only, **no L1** | read sentences | transcripts, alignable | CC0 — [site](https://commonvoice.mozilla.org) | **yes** | free bulk |
| GLOBE | 535 h curated | 23,519 | native/nativized accents | read | transcripts + accent/age | CC BY 4.0 — [arXiv](https://arxiv.org/abs/2406.14875) | yes | free |
| Wildcat Corpus | (UNVERIFIED) | 84 | zh, hi/mr, … | scripted + **spontaneous** | orthographic transcripts | UNVERIFIED — [SpeechBox](https://speechbox.linguistics.northwestern.edu/wildcat-scripted) | UNVERIFIED | request |
| TLT-school | 49 h EN | Italian children | it (children) | read | transcripts + CEFR | CC BY-NC 4.0 — [LREC](https://aclanthology.org/2020.lrec-1.47) | no (NC) | shared task |
| CSLU Foreign Accented EN (LDC2007S08) | ~5 h (UNVERIFIED) | ~150 (UNVERIFIED) | 22 incl. **ru** | telephone read | transcripts | LDC license — [catalogue](https://catalog.ldc.upenn.edu/LDC2007S08) | no | paid (LDC) |
| CU-CHLOE | UNVERIFIED | UNVERIFIED | yue (zh) | words/sentences/paragraphs | L2 mispronunciation variants | UNVERIFIED; not in LDC — [paper](https://www1.se.cuhk.edu.hk/~hccl/publications/pub/MengAPSIPA2010.pdf) | UNVERIFIED | restricted (CUHK) |
| ERJ (UME-ERJ) | UNVERIFIED | ~200 | ja | read | phonetic transcriptions | NII-SRC / ELRA — [site](https://research.nii.ac.jp/src/en/UME-ERJ.html) | no | registration |
| EdAcc (Edinburgh) | ~40 h | 122 | 40+ accents | **spontaneous dyads** | alignable transcripts | CC BY-SA 4.0 — [datashare](https://datashare.ed.ac.uk/handle/10283/4766) | conditional (BY-SA) | free |
| AccentDB | small (UNVERIFIED) | tens | 8 (4 Indian EN) | read | transcripts | CC BY-NC 4.0 — [accentdb.org](https://accentdb.org) | no (NC) | free |
| Sell-corpus | UNVERIFIED | UNVERIFIED | zh dialects | read | **phoneme substitution annotations** | UNVERIFIED — [poster](https://sigport.org/sites/default/files/docs/sell-corpus_poster_0.pdf) | UNVERIFIED | ECNU authors |
| NICT-JLE | ~300 h spoken | 1,281 | ja | spontaneous | error tags; **no audio** | NICT — [site](https://alaginrc.nict.go.jp/nict_jle/index_E.html) | n/a | free (text) |

### 4.2 The four reality answers

**(a) Largest openly available non-native resource, and usable /ð/ /θ/ hours.**
Largest *open* is Common Voice EN (~3,000+ h, CC0) — but it is **not
non-native**: accent tags are native/nativized English regions, there is **no
L1 metadata**, and no error labels. The largest explicitly non-native corpus
is AESRC2020 (160 h) — commercially gated, not downloadable. The largest
*openly downloadable* corpus with phone-error annotations is **ISLE at
17 h 54 m** (2 L1s). Realistic usable error-labeled /ð/+/θ/ data across
L2-ARCTIC + SpeechOcean762 + ISLE + EpaDB + Sell-corpus: **a few hundred to
low thousands of phone tokens — minutes of audio, not hours.** Force-aligning
a CC0 corpus yourself yields thousands of /θ/ /ð/ *tokens*, but with no error
labels and no L1 — usable for correct-class contrastive features, not for a
supervised error detector.

**(b) Any large phone-error corpus beyond ours?** No. The complete open
inventory of phone-error-annotated non-native English is small: ISLE
(17 h 54 m), L2-ARCTIC (~27 h, 24 spk), SpeechOcean762 (~10 h), EpaDB, 
Sell-corpus, CU-CHLOE (restricted). ISLE is the largest and it covers two L1s.

**(c) Russian-accented English specifically.** The only substantial labeled
set is the **AESRC2020 RU subset (~20 h, ~60 speakers)** — Datatang-commercial
and gated. CSLU Foreign Accented English (LDC2007S08) includes Russian among
22 L1s but only a small slice (~150–250 utterances), telephone audio.
Common Voice "Russian speakers' English clips" **cannot be extracted**: CV
records no native-language field and ru/en clips are not linked. No dedicated
open Russian-L1 English *audio* corpus exists (the Russian learner corpora
are text-only). Net: Russian audio requires either pursuing Datatang/LDC
access or self-collection.

**(d) Synthetic data, license-clean (one paragraph).** Use a permissively
licensed model driven by CC0/CC-BY reference audio. Avoid XTTS-v2 (CPML
weights) and MMS-TTS (CC-BY-NC). Clean options: **Kokoro** (Apache-2.0/MIT),
**MeloTTS** (MIT, nativized accents), **StyleTTS2** (MIT), **CosyVoice**
(Apache-2.0, zero-shot cloning + cross-lingual — the best for accented L2
samples), and **RVC** voice conversion (MIT). The clean recipe: (1) source
reference L2 audio from CC0/CC-BY (Common Voice, GLOBE, our CC-BY
SpeechOcean762); (2) fine-tune a MIT/Apache TTS/VC model on it; (3) synthesize
read sentences/paragraphs. Caveats: the reference audio's license propagates
to the cloned model, and synthetic mispronunciations are **not guaranteed
phonetically faithful** — validate against real L2 data before trusting them
for judging.

## 5. Feasibility verdict: "am I demanding too much?"

**One-line answer: you are not demanding too much of the *models* — but you
are asking for a dataset that does not exist in open form. Redefine "huge":
scale comes from CC0 mass + synthetic audio; precision comes from the small
real error corpora we already hold.**

Three sub-verdicts, with the numbers:

1. **Training a better judge — REALISTIC.** Transfer learning, not from
   scratch: frozen XLS-R-300M + linear/MLP head runs on 1.5–2.5 GB VRAM
   (even the Mac can train the head), and the 2025 MDD literature shows
   frozen SSL features + shallow classifiers already work (arXiv 2511.20107).
   Our data supports it: 2,370 real error tokens across /ð/+/θ/. The
   remaining risk is not data volume but the thin /z/-realization class
   (93 positives) — handled by per-substitution binary heads + oversampling.
2. **Training from scratch on huge data — NOT REALISTIC, and unnecessary.**
   Pretraining an encoder on even 160 h on an 8 GB laptop GPU is a
   non-starter, and it is the wrong instrument anyway: the literature and
   M1's own numbers say the problem is the *head*, not the representation.
   Fine-tuning beats scratch at every data size we can reach.
3. **Obtaining huge *real* non-native corpora — DOESN'T EXIST OPENLY.** The
   open ceiling for error-annotated L2 audio is ~18 h (ISLE) and it is
   2-L1, NC-licensed; the one 160 h non-native set (AESRC2020) is
   commercially gated; Common Voice's 3,000+ h has no L1 metadata.
   "Huge" is achievable only as a *construction*: CC0 Common Voice mass
   (correct class) + our small real error sets + SAA/ISLE as eval anchors +
   synthetic accented speech (§4.2d).

So the honest reading of the original demand: **keep the ambition, change the
recipe.** The connected-speech gap is real (error labels live in read
sentences; spontaneous L2 audio has no error labels — EdAcc/Wildcat are
label-free), so connected-speech training must come from paragraph-style
synthesis + alignment, while real error labels stay segment-sourced from the
read corpora.

## 6. Recommended experiment ladder

Ranked by expected value per GPU hour, each reusing the existing eval
harness and the speaker-disjoint discipline that caught M1's leakage. No bar
loosening: P ≥ 0.90 @ R ≥ 0.40, per-L1 audit, ~30-item human spot-check.
Commands below match the repo's actual CLIs (verified 2026-08-20).

**Exam command (any candidate model):**

```bash
cd eval && uv run python run_eval.py \
    --contrast "ð:z,d,v" \
    --l2arctic data/l2arctic --speechocean762 data/speechocean762 \
    --model <candidate-id> \
    --out reports/
```

1. **Frozen XLS-R-300M + linear probe (THE next experiment).** Pool layers
   12–20 over the aligned /ð/ (later /θ/) segments; logistic regression /
   linear SVM on the 1024-d features; train on the L2-ARCTIC train split,
   exam on held-out speakers. Hours of work, runs on the Mac, and directly
   tests whether M1's CTC head was the bottleneck. Needs a small new
   training script (current training/train.py is charsiu/wav2vec2-CTC-
   specific); the candidate drops into the registry as a
   role="contrast" ModelSpec with repo_id="local" and takes the exam through
   the same --model path as the Option 3 judges.
2. **Classical acoustic baseline.** Fricative spectral moments (center of
   gravity, skewness), voicing fraction, duration → GBDT/SVM. The cheap
   control that answers "did we even need deep learning for /ð/ vs /z/?"
   — especially relevant given the dental-fricative literature is largely
   acoustic, not classifier-based.
3. **Fine-tune / LoRA** on combined L2 data if 1–2 fall short. Current recipe
   pattern (verified flags):
   ```bash
   uv run python -m openschwa_training.train \
       --data data/l2arctic-dh --base-model ../.models/<base> \
       --out runs/v1 --epochs 12 --freeze-epochs 4 --batch-size 32 \
       --alphabet "ð,z,d,v" --label-smoothing 0
   ```
   (also available: --lr-head, --lr-full, --target-boost, --hardneg-mult,
   --augment, --warmup-frac, --seed, --max-steps)
4. **Synthetic-data pipeline** (Kokoro/CosyVoice + rule-based error
   injection, §4.2d) — only after 1–3 show which model family deserves the
   scale.

## 7. Open questions for the project owner

To be answered before any training spend (numbered so they can be answered in
chat as "1: yes, 2: no, 3: …"):

1. **Scope:** do we re-attack /ð/ only (M1 parity), or /ð/ + /θ/ together
   from the start? (§4.0 shows /θ/ is well supported: 397 error tokens.)
2. **Russian priority:** Russian-accented audio exists only as gated
   AESRC2020 RU (~20 h, commercial) or a tiny LDC slice. Do we (a) pursue
   Datatang access, (b) self-record teacher-curated RU speakers, or (c) defer
   RU and keep per-L1 fairness across the six existing groups as the gate?
3. **Paid corpora:** ISLE is free with ELRA registration (non-commercial
   terms); LDC items are paid. Is any budget available, or is free +
   registration-only the hard rule?
4. **Weight-distribution policy:** NC-licensed corpora (L2-ARCTIC, SAA
   mirror, ISLE) can train a model, but distributing the resulting weights is
   legally murky. Do we accept "eval-only" for those and train shippable
   weights only on CC0/CC-BY data (Common Voice, SpeechOcean762, synthetic),
   or is a purely local, never-distributed judge acceptable?
5. **Hardware ceiling:** the frozen-head recipe runs on the Mac; LoRA needs
   the 4060. Is that the ceiling, or is one-off cloud GPU rental conceivable?
   (Only changes step 3's model tier, not steps 1–2.)

## Sources (primary)

Model cards and LICENSE files are linked inline in §3; corpus sources inline
in §4; papers inline in §3.5. Items marked UNVERIFIED could not be confirmed
from a primary source during this research pass — verify before depending on
them.
