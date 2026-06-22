# SpectralBridge

**A first research attempt at aligning audio and images directly, without a language bridge.**

GitHub: github.com/alperarslan19/spectralbridge

This started as an experiment and ended as a lesson about its own limits. I'm keeping the results and the numbers here, but the most useful outcome was seeing — once I looked closely — that the setup could not actually answer the question I started with. This README is written from that perspective: what I tried, what the numbers show, where the setup falls short, and what the real question would require.

## The question

Most models that generate images from sound (or sound from images) connect the two through text-mediated encoders like CLIP and CLAP — they relate audio and images through a shared semantic space shaped by language. That is very effective for high-level, conceptual matching ("there is a guitar in the sound, there is a guitar in the image"), but it may lose the more direct, more textural relationships between a sound and an image.

So the question was: can audio and images be aligned **directly**, through the structure already present in each modality's own representation, without routing through language?

## What I did

I took 1,269 audio–image pairs from a 16-category subset of [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) and passed each one through four **frozen** pretrained encoders:

- **language-free side:** Audio-MAE (768-d) and DINOv2 (768-d)
- **text-mediated side:** CLAP (512-d) and CLIP (512-d)

The embeddings were cached once (`02_feature_extraction.ipynb`), so every later experiment runs only on cached vectors — fast to train, and every method sees identical inputs.

I then trained small **bridge** models between these frozen representations with a contrastive objective (InfoNCE), so that the correct audio–image pairs end up close in the shared space and incorrect pairs far apart. Three bridge types: a **linear probe**, a **ReLU MLP**, and a **frequency-based design** (the original proposal, discussed below). The design lets two factors be separated: the **encoder pair** (language-free vs text-mediated) and the **bridge** (linear vs nonlinear).

## Results

Audio→image retrieval on a held-out validation set (254 pairs, gallery = val images only):

| Method | Encoders | Trainable params | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| **Linear probe** | Audio-MAE / DINOv2 | 590,592 | **17.32** | **45.67** | 62.20 |
| Linear probe | CLAP / CLIP | 262,656 | 13.39 | 37.80 | 61.02 |
| Trained MLP | CLAP / CLIP | 131,712 | 14.96 | 36.22 | 57.87 |
| ReLU MLP | Audio-MAE / DINOv2 | 214,016 | 11.81 | 34.25 | 56.30 |
| Frequency-based bridge | Audio-MAE / DINOv2 | 132,096 | 0.39 | 2.76 | 5.91 |
| Zero-shot CLAP→CLIP | CLAP / CLIP | 0 | 0.79 | 2.36 | 5.51 |
| *Random chance* | — | — | *0.39* | *1.97* | *3.94* |

Three things stand out, and the interpretation matters more than the numbers:

**1. The simplest bridge works best.** A single linear map between the two frozen embedding spaces reaches 45.7% R@5 (chance: 2.0%) and beats the same-capacity nonlinear MLP. The reading I'd give now: two encoders trained independently, on different modalities, with no shared supervision, still share enough common semantic structure to be aligned by a simple linear transform. Added complexity in the bridge doesn't help — which already hints that what's being aligned here is fairly coarse.

**2. Neither encoder family is clearly better.** With a linear bridge the language-free pair leads (45.7% vs 37.8%); with an MLP bridge the text-mediated pair edges ahead (36.2% vs 34.3%). Both gaps are small, and the direction flips depending on the bridge — so "language-free aligns better," my original expectation, is **not** supported once the comparison is made symmetric.

**3. Text-mediated encoders transfer poorly zero-shot but adapt cheaply.** Zero-shot CLAP→CLIP is at chance (2.4% R@5); a small trained projection lifts it to 36–38%.

## The main lesson: the setup couldn't test the original question

The original idea was that language-free encoders might capture **finer, more direct** audio–image relationships than text-mediated ones. Looking closely, the setup was not suited to test that — at several levels.

**Data.** VGGSound is built from YouTube videos, and the audio and image in a pair are often not from the same moment, recorded together. A large fraction are mismatched, animated, or have a static cover image over unrelated audio. A relationship can only be learned if it exists in the data, and here the data mostly supports **category-level** matching, not the fine, instance-level relationship I was after.

**Loss.** With in-batch negatives drawn from across categories, a bridge can minimize the contrastive loss just by telling categories apart ("this is wind, not piano") — it is never pushed to learn the finer within-category relationship. The category shortcut is enough.

**Metric.** Retrieval (Recall@K) only asks whether the correct image ranks near the top. It does not separate a hit that comes from **category-level** matching ("wind audio → the wind-image region") from one that comes from **instance-level** matching ("this wind clip → its own frame"). With a multi-category gallery, knowing the category alone already scores well.

These three limits point the same way: the experiment measured **category-level alignment**, and the question I actually cared about — the finer, direct relationship — was never really put to the test, neither confirmed nor ruled out.

## The frequency-based bridge

The original proposal — the one the repo is named after — was a bridge using Fourier feature mapping and SIREN (sine) activations. The motivation was a question rather than a claim: since a spectral relationship between sound and image might, in principle, leave some trace inside the embeddings, could a frequency-aware bridge pick it up?

It didn't. It stayed at chance level (2.8% R@5), below even the standard MLP. A same-capacity ReLU MLP — identical depth, width, dropout, data, and training budget, differing only in the Fourier+SIREN block — generalized fine (34.3%). So this is a clean ablation: the failure is localized specifically to that block, not to capacity, data, or training.

Why it failed, as I read it now:

- The frozen random 768→64 projection discards most of the embedding's information irreversibly. A *learned* 768→128 compression (as in the MLP) can choose what to keep; a random one cannot.
- Fourier feature mappings and SIREN were designed for **low-dimensional coordinate inputs** (e.g. pixel positions), where the goal is to represent fine detail of a function over a continuous domain. Applied to a 768-dimensional, abstract, already-compressed embedding, the assumptions they rely on don't hold — there's no reason to expect periodic structure in that kind of representation in the first place.

I don't read this as "the frequency idea is wrong" — the architecture failed at the basic categorical level, so it never reached the point where a finer hypothesis could even be tested. It's better described as the wrong tool for this representation.

## What a better version would need

Two things, both of which this project made concrete for me:

**Cleaner, genuinely synchronized data** — and not just "from the same video." The ideal is the moments of a *single continuous event* synchronized over time: e.g. the sound and image of each moment of a speech, frame by frame. The direct relationship lives there, in the sound and image of the *same instant* — exactly what a single middle-frame-per-clip setup throws away.

**End-to-end training.** Instead of bridging two frozen, already-compressed embeddings after the fact, the two encoders should be trained together, so the gradient flows through both modalities from the start. If the fine information is discarded by the time we reach the cached embeddings, no bridge trained afterward can recover it.

## Discussion

The strongest result — a plain linear map aligning two independently trained encoders — connects to a broader question about representation convergence (cf. the Platonic Representation Hypothesis): encoders that never saw each other's modality still appear to organize the world along largely compatible axes, differing mainly by a coordinate transform.

But the experiment **cannot distinguish** category-level alignment from instance-level alignment. With 16 categories and ~16 validation examples per category, a bridge that only learned categories could already score well at R@5. Separating the two readings would require **within-category retrieval** (restricting the gallery to a single category, so category cues are useless and only a finer relationship can rank the correct item), or probing the embeddings for a continuous physical factor (e.g. wind intensity). Both are left as future work, along with scaling beyond 1,269 pairs.

## A note on method corrections

Two flaws in earlier versions of this experiment were found and fixed; they are documented rather than hidden because they changed the headline numbers substantially:

- **Evaluation leakage.** An earlier evaluation used all 1,269 pairs as both queries and gallery, letting training samples into the gallery and inflating scores for memorizing models. Fixed: gallery = the 254 validation images only.
- **Under-training.** Because a cached-feature "epoch" is only ~2 gradient steps, an earlier budget (100 epochs, patience 15) stopped most bridges after ~30–40 steps. An apparent large gap between encoder pairs was largely an under-training artifact; with a proper budget (up to 1,000 epochs, patience 100) the encoder pairs are roughly on par.

## Method details

All four encoders are frozen; embeddings are extracted once and cached. Bridges are trained with InfoNCE (temperature 0.07, batch size 512 → 511 in-batch negatives), outputs L2-normalized so training and cosine-similarity retrieval share the same unit-sphere geometry. Optimizer AdamW (lr 1e-4, weight decay 0.01), cosine LR annealing, gradient clipping, early stopping on validation loss.

Bridge architectures:
- **Linear probe** (d→d): a full-rank linear map, no bottleneck, no nonlinearity. Run for both encoder pairs (768→768 and 512→512).
- **ReLU MLP** (768→128→128→768): same capacity and bottleneck as the frequency-based bridge but with standard activations — isolates the contribution of the Fourier+SIREN block.
- **Frequency-based bridge** (768 → frozen random Fourier features (64-d, σ=10) → 2 SIREN layers (128-d) → 768).
- **CLAP→CLIP MLP** (512→128→512): nonlinear bridge for the text-mediated pair.

## Dataset

1,269 audio–image pairs from a 16-category subset of VGGSound: the audio track and a middle video frame of each 10-second clip. Split 80/20 (1,015 train / 254 val) with a fixed seed. Categories span both "textually easy" sounds (instruments, animals) and perceptually richer environmental textures (wind, rain, fire, water).

## Repository structure

```
notebooks/
  01_download_vggsound.ipynb           # dataset download (yt-dlp, VGGSound subset)
  02_feature_extraction.ipynb          # frozen encoders → cached .npy embeddings
  03_train_and_evaluate_bridges.ipynb  # bridges, baselines, evaluation
```

## Reproducibility

All training runs in `03_train_and_evaluate_bridges.ipynb` are seeded (`torch.manual_seed(42)`, `np.random.seed(42)`) and the full notebook runs end-to-end on a single Colab T4 in well under an hour, given the cached embeddings. The train/val split uses a fixed `RandomState(42)` permutation shared by all methods. Exact retrieval numbers may vary by a small margin across runs due to GPU non-determinism; reported numbers are from a single clean end-to-end run, matching the committed notebook outputs.
