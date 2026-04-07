#!/usr/bin/env python3
"""
Daily ML blog post generator for sadjadalikhani.github.io
Uses the OpenAI API to write rich, visually stunning Jekyll posts
about cutting-edge ML / foundation-model research.

Usage:
    python3 generate_post.py             # generate + git commit + push
    python3 generate_post.py --dry-run   # write file locally, skip git
    python3 generate_post.py --force     # overwrite today's post if it exists
"""

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
SITE_DIR   = Path(os.environ.get("SITE_DIR", str(BASE.parent / "website" / "sadjadalikhani.github.io")))
STATE_FILE = BASE / "state.json"
LOG_FILE   = BASE / "publisher.log"
CONFIG_ENV = BASE / "config.env"

# ── Load config.env ─────────────────────────────────────────────────────────
if CONFIG_ENV.exists():
    for _line in CONFIG_ENV.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    sys.exit("ERROR: OPENAI_API_KEY not set. Add it to blog-publisher/config.env")


# ─────────────────────────────────────────────────────────────────────────────
#  TOPICS  (20 entries, cycling forever)
#  youtube: a confirmed real video ID for each topic.
# ─────────────────────────────────────────────────────────────────────────────
TOPICS = [
    {
        "title": "Flash Attention: Making Transformers Faster Than Ever",
        "slug":  "flash-attention",
        "tags":  "attention transformers efficiency hardware",
        "cat":   "foundation-models",
        "desc":  "A deep dive into Flash Attention — the IO-aware exact attention algorithm that makes training large language models dramatically faster while using far less memory.",
        "yt":    "gMOAud7hZg4",
        "focus": """
Flash Attention 1, 2, and 3 (Dao et al.).
Cover: O(N²) memory wall of standard attention, HBM bandwidth bottleneck,
tiling trick to stay within SRAM, backward pass via recomputation,
causal masking integration, multi-head variant.
ECharts bar chart: throughput (tokens/sec) for Standard vs FA1 vs FA2 vs FA3
at sequence lengths 512 / 1 k / 2 k / 4 k / 8 k / 16 k.
Idiomatic PyTorch pseudo-code for the tiled forward kernel (< 35 lines).
Full math: why tiling preserves exact softmax — the log-sum-exp online trick.
Mention: xFormers, torch.nn.functional.scaled_dot_product_attention, FlexAttention.""",
    },
    {
        "title": "Mixture of Experts: Scaling AI Without Breaking the Bank",
        "slug":  "mixture-of-experts",
        "tags":  "moe scaling llm efficiency sparse",
        "cat":   "foundation-models",
        "desc":  "How Mixture-of-Experts architectures let language models reach trillion-parameter scale while keeping per-token compute tractable.",
        "yt":    "UUs4DF5lFyw",
        "focus": """
Sparse MoE: top-k routing, softmax gating, load-balancing auxiliary loss.
Switch Transformer (single expert), GLaM, Mixtral 8×7B, DeepSeek-MoE.
Expert collapse problem, expert specialization patterns.
ECharts radar chart: MoE vs dense models across total params / active params /
throughput / memory / MMLU score (5 axes).
Math: g(x) = softmax(W_g x), top-k selection, L_aux load-balancing loss.
PyTorch: minimal MoE FFN layer with top-2 routing.""",
    },
    {
        "title": "Mamba and State Space Models: The Sequence Modelling Revolution",
        "slug":  "mamba-state-space-models",
        "tags":  "ssm mamba recurrence linear sequence",
        "cat":   "foundation-models",
        "desc":  "State Space Models and Mamba's input-selective mechanism — linear-time sequence modelling that rivals Transformers on long sequences.",
        "yt":    "9dSkvxS2EB0",
        "focus": """
S4 (continuous SSM, ZOH discretization, HiPPO init).
Mamba: input-dependent Δ, B, C — content-aware unlike S4's LTI.
Hardware-aware parallel scan, O(N) training.
Mamba-2 and SSD (Structured State Space Duality).
ECharts line chart (log-log): wall-clock time/token vs sequence length 1 k–100 k
for Transformer (O(N²)) vs Mamba (O(N)).
Math: h_t = Ā h_{t-1} + B̄ x_t, y_t = C h_t + D x_t; SSM convolution view.
PyTorch: selective SSM scan stub with annotated shapes.""",
    },
    {
        "title": "RLHF and DPO: Teaching Language Models to Be Helpful and Harmless",
        "slug":  "rlhf-dpo-alignment",
        "tags":  "rlhf dpo alignment safety preference",
        "cat":   "alignment",
        "desc":  "The complete alignment pipeline — from SFT to RLHF with PPO, to Direct Preference Optimization that eliminates the reward model entirely.",
        "yt":    "hhiLw5Q_UFg",
        "focus": """
Three-stage RLHF: SFT → reward model → PPO with KL penalty.
Reward hacking, Goodhart's Law. Constitutional AI (Anthropic): self-critique + RLAIF.
DPO: closed-form solution, log-ratio preference objective, equivalence to RLHF.
ECharts scatter: Pareto front of helpfulness vs harmlessness for
SFT / RLHF-PPO / DPO / CAI on MT-Bench × TruthfulQA grid.
Math: PPO clip objective + KL, DPO loss = -log σ(β [log π_θ/π_ref (chosen) - log π_θ/π_ref (rejected)]).
PyTorch: DPO training step using HuggingFace TRL.""",
    },
    {
        "title": "Diffusion Models: The Probabilistic Engine Behind Generative AI",
        "slug":  "diffusion-models-deep-dive",
        "tags":  "diffusion generative score-matching ddpm",
        "cat":   "generative-ai",
        "desc":  "A rigorous but accessible walkthrough of DDPM, score matching, and latent diffusion — the mathematical backbone of Stable Diffusion and DALL·E.",
        "yt":    "fbLgFrlTnGU",
        "focus": """
DDPM forward process q(x_t | x_{t-1}), reparameterized ELBO, noise prediction ε_θ.
Noise schedules: linear, cosine. DDIM fast sampling (50 vs 1000 steps).
Classifier-free guidance. Latent Diffusion (Stable Diffusion): VAE + U-Net in latent space.
Flow Matching as a cleaner generalization.
ECharts bar chart: FID on CIFAR-10 for DDPM / DDIM-50 / LDM / Flow Matching.
Math: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I), ELBO decomposition, ε-prediction objective.
PyTorch: DDPM forward diffusion + reverse sampling loop.""",
    },
    {
        "title": "Vision Transformers: How Attention Conquered Computer Vision",
        "slug":  "vision-transformers-vit",
        "tags":  "vit vision patches self-supervised dino mae",
        "cat":   "foundation-models",
        "desc":  "From patch embeddings to DINOv2 — the complete story of how Transformers revolutionized computer vision.",
        "yt":    "TrdevFK_am4",
        "focus": """
ViT: image → non-overlapping patches → linear embedding + class token + 1D pos enc → Transformer encoder.
DeiT (data-efficient ViT with distillation token).
DINO / DINOv2: self-distillation, emergent segmentation in attention maps.
MAE: 75% mask ratio, asymmetric encoder-decoder, linear probing results.
ECharts bar chart: ImageNet top-1 accuracy for
ResNet-50 / ResNet-152 / ViT-S/16 / ViT-B/16 / ViT-L/16 / DINOv2-L.
Math: x_patch = flatten(P_i) W_E + b; attention rollout for visualization.
PyTorch: complete ViT forward pass in < 30 lines.""",
    },
    {
        "title": "RoPE and ALiBi: Giving Transformers Unlimited Memory",
        "slug":  "rotary-positional-embeddings-rope",
        "tags":  "rope positional-encoding long-context transformers",
        "cat":   "foundation-models",
        "desc":  "How RoPE, ALiBi, and YaRN enable language models to handle context windows from 4 k to over 1 million tokens.",
        "yt":    "o29P0Kpobz0",
        "focus": """
Absolute sinusoidal PE, relative PE (T5 bias), learned PE — extrapolation failures.
ALiBi: linear bias on attention logits, length extrapolation without retraining.
RoPE: rotate q and k by position-dependent angle; complex-number formulation;
dot product decays monotonically with |i-j|.
YaRN and LongRoPE: rescaling θ-base frequencies to extend context cheaply.
ECharts heatmap: attention score decay with relative distance |i-j| (0–512)
for absolute PE / sinusoidal / ALiBi / RoPE — 4 rows, colour-coded.
Math: RoPE(q, m) = R^d_{Θ,m} q where R rotates 2-D subspaces by m θ_i.
PyTorch: RoPE applied to Q and K matrices.""",
    },
    {
        "title": "LoRA and QLoRA: Fine-Tuning 70 B Models on a Consumer GPU",
        "slug":  "lora-parameter-efficient-finetuning",
        "tags":  "lora qlora peft fine-tuning efficiency",
        "cat":   "efficiency",
        "desc":  "LoRA, QLoRA, and the PEFT ecosystem — how the intrinsic dimensionality hypothesis lets us fine-tune billion-parameter models on a single GPU.",
        "yt":    "dA-NhCtrrVE",
        "focus": """
Full fine-tuning cost, intrinsic dimensionality hypothesis.
LoRA: W = W_0 + BA, rank r << d, scaling α/r, init B=0.
QLoRA: 4-bit NF4 + double quantization + paged AdamW + LoRA.
Adapter layers, prefix tuning, prompt tuning comparison on MMLU.
DoRA: decompose weight into magnitude × direction.
ECharts bubble chart: MMLU accuracy vs GPU memory at training;
bubble size = trainable parameters.
Points: Full FT / LoRA-r8 / LoRA-r64 / QLoRA-r64.
Math: ΔW = BA; total trainable params = 2rd; gradient flows only through A and B.
PyTorch: applying LoRA to attention layers with HuggingFace PEFT.""",
    },
    {
        "title": "Retrieval-Augmented Generation: Grounding LLMs in Facts",
        "slug":  "retrieval-augmented-generation",
        "tags":  "rag retrieval llm vector-search knowledge",
        "cat":   "foundation-models",
        "desc":  "How RAG systems combine dense vector retrieval with language model generation to produce factually grounded, up-to-date answers.",
        "yt":    "T-D1OfcDW1M",
        "focus": """
Naive RAG: chunk → embed → ANN store → retrieve → generate.
Dense retrieval: bi-encoder (DPR), cross-encoder reranking, ColBERT late interaction.
Advanced RAG: query rewriting, HyDE, multi-hop, self-RAG.
Chunking strategies: fixed-size, semantic, recursive.
ECharts stacked bar: end-to-end latency breakdown
(embed / ANN search / rerank / LLM) at corpus size 100 / 1 k / 10 k docs.
Math: cosine score sim(q, d) = q·d / (||q|| ||d||); NDCG evaluation.
Python: end-to-end RAG in 35 lines — sentence-transformers + FAISS.""",
    },
    {
        "title": "Chain-of-Thought: Why Thinking Out Loud Makes AI Smarter",
        "slug":  "chain-of-thought-reasoning",
        "tags":  "cot reasoning prompting self-consistency o1",
        "cat":   "foundation-models",
        "desc":  "Chain-of-thought prompting, self-consistency, Tree-of-Thoughts, and the new era of reasoning models that scale test-time compute.",
        "yt":    "iyioi2MJdgU",
        "focus": """
Standard prompting vs CoT (Wei et al. 2022). Few-shot vs zero-shot CoT.
Self-consistency: sample N chains, majority vote — why it works probabilistically.
Tree-of-Thoughts: BFS/DFS over reasoning steps with a value function.
Process reward models (PRM) vs outcome reward models (ORM).
OpenAI o1 / o3, DeepSeek-R1: extended thinking, inference-time compute scaling.
ECharts grouped bar: GSM8K accuracy for GPT-3.5 / GPT-4 across
standard / few-shot CoT / zero-shot CoT / self-consistency (N=40).
Math: P(a|x) ≈ Σ_r P(a|r, x) P(r|x) — marginalizing over reasoning chains.
Python: self-consistency with temperature sampling and majority-vote aggregation.""",
    },
    {
        "title": "Neural Scaling Laws: The Power Laws Governing Every LLM",
        "slug":  "neural-scaling-laws",
        "tags":  "scaling laws compute llm chinchilla kaplan",
        "cat":   "foundation-models",
        "desc":  "Kaplan's and Chinchilla's scaling laws demystified — the power laws every major LLM training run is designed around.",
        "yt":    "UFSFMEgQJic",
        "focus": """
Kaplan et al. (2020): L(N) and L(D) power laws, compute-optimal frontier.
Chinchilla (Hoffmann 2022): 20 tokens/parameter; why GPT-3 was undertrained.
Three-way Pareto: N, D, C = 6ND.
Emergent abilities — discontinuous jumps or metric artefact?
Inference-time scaling: o1/o3, trading training compute for test-time compute.
ECharts scatter (log-log): validation loss vs model parameters,
power-law fit line, marking GPT-2 / GPT-3 / Chinchilla / LLaMA-3.
Math: L(N,D) = E + A/N^α + B/D^β; optimal N*(C) ∝ C^0.5, D*(C) ∝ C^0.5.
Python: fitting a power law with scipy.optimize.curve_fit on synthetic data.""",
    },
    {
        "title": "Multimodal Foundation Models: Teaching AI to See and Read Together",
        "slug":  "multimodal-foundation-models",
        "tags":  "multimodal clip llava vision-language flamingo",
        "cat":   "foundation-models",
        "desc":  "CLIP, LLaVA, Flamingo, and GPT-4V — how modern AI systems fuse vision and language into unified world representations.",
        "yt":    "T9XSU0pKX2E",
        "focus": """
CLIP: contrastive image-text pretraining, InfoNCE loss, zero-shot transfer.
Flamingo: perceiver resampler + cross-attention into frozen LLM.
LLaVA: linear projection maps CLIP ViT features into LLM token space.
InstructBLIP, SigLIP, GPT-4V, Gemini 1.5, Claude 3 multimodal.
ECharts grouped bar: zero-shot ImageNet top-1 accuracy for
CLIP ViT-B/32 / B/16 / L/14 / OpenCLIP-H/14 / SigLIP-L/16.
Math: InfoNCE L = -Σ_i log exp(sim(z_i, z'_i)/τ) / Σ_j exp(sim(z_i, z'_j)/τ).
Python: CLIP zero-shot classification in 20 lines.""",
    },
    {
        "title": "Sparse Autoencoders: The Dictionary of Concepts Inside LLMs",
        "slug":  "sparse-autoencoders-llm-features",
        "tags":  "sae interpretability features superposition",
        "cat":   "interpretability",
        "desc":  "How sparse autoencoders are helping researchers discover millions of monosemantic features inside large language models — a breakthrough in AI interpretability.",
        "yt":    "9BPCJYQ3KAA",
        "focus": """
Superposition hypothesis: networks store more features than dimensions.
SAE architecture: encoder → ReLU → decoder (unit-norm columns).
Training: MSE reconstruction + L1 sparsity penalty λ.
Anthropic's paper: 34 M features in Claude 3 Sonnet residual stream.
Top-k SAEs: hard k-sparse activations replacing L1 (cleaner, no shrinkage).
Feature geometry: uniform norms, antipodal pairs, privileged basis.
ECharts scatter (log-log): feature activation frequency vs mean activation value
showing power-law distribution of feature usage.
Math: f(x) = ReLU(W_e (x - b_d) + b_e),
      L = ||x - W_d f(x) - b_d||_2^2 + λ ||f(x)||_1.
PyTorch: minimal top-k SAE training loop on GPT-2 residual stream.""",
    },
    {
        "title": "Speculative Decoding: 3× Faster LLM Inference for Free",
        "slug":  "speculative-decoding",
        "tags":  "inference efficiency speculative-decoding latency",
        "cat":   "efficiency",
        "desc":  "How speculative decoding uses a small draft model and one parallel verification pass to dramatically accelerate autoregressive inference.",
        "yt":    "eqOfr7-_S8s",
        "focus": """
Autoregressive bottleneck: LLMs are memory-bandwidth limited, not compute limited.
Speculative decoding: draft γ tokens with small model → verify all in one LLM forward pass.
Acceptance criterion: accept token x if p_large(x) / p_draft(x) ≥ Uniform[0,1].
This preserves the large-model distribution exactly.
Medusa: multiple FFN draft heads from the same backbone.
EAGLE / EAGLE-2: drafting in feature space for higher acceptance rates.
ECharts grouped bar: tokens/sec for Standard / Spec-γ3 / Spec-γ5 / Medusa / EAGLE
across 7B / 13B / 70B model families.
Math: E[accepted] = (1 - α^{γ+1}) / (1 - α), α = mean token acceptance rate.
Python: speculative decoding sampling loop from scratch.""",
    },
    {
        "title": "Mechanistic Interpretability: Reverse-Engineering the Transformer",
        "slug":  "mechanistic-interpretability",
        "tags":  "interpretability circuits induction-heads features",
        "cat":   "interpretability",
        "desc":  "How researchers use circuits, activation patching, and the logit lens to understand exactly what computations happen inside Transformer models.",
        "yt":    "KuXjwB4LzSA",
        "focus": """
Circuits hypothesis: networks implement human-interpretable algorithms in subgraphs.
Induction heads: the key circuit behind in-context learning.
Copy suppression heads, indirect object identification (Wang et al.).
Causal tracing / activation patching (ROME): locating factual associations.
Logit lens: interpreting intermediate residual stream states by projecting to vocabulary.
ECharts heatmap (12×12): attention pattern of a model induction head —
strong off-diagonal band at position +1.
Math: residual stream x_L = x_0 + Σ_l attn_l + Σ_l mlp_l;
direct logit attribution via unembedding.
Python: activation patching with TransformerLens to locate a factual circuit.""",
    },
    {
        "title": "The Transformer Architecture: A First-Principles Deep Dive",
        "slug":  "transformer-architecture-deep-dive",
        "tags":  "transformers attention architecture foundational",
        "cat":   "foundation-models",
        "desc":  "A rigorous technical walkthrough of every sublayer in the original Transformer — the architecture underpinning virtually all modern AI.",
        "yt":    "iDulhoQ2pro",
        "focus": """
Encoder-decoder full architecture: multi-head self-attention + cross-attention + FFN.
Scaled dot-product: Attention(Q,K,V) = softmax(QK^T / √d_k) V.
Multi-head: h independent heads, concatenate, project W_O.
Pre-LN vs Post-LN residual connections — training stability implications.
Sinusoidal positional encoding: enables relative attention via dot products.
BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder) trade-offs.
ECharts heatmap (12×12 tokens): realistic attention weight matrix with warm colour scale.
Math: full MHA formula; FFN(x) = max(0, xW_1 + b_1)W_2 + b_2.
PyTorch: complete single self-attention block in 25 lines.""",
    },
    {
        "title": "Contrastive Self-Supervised Learning: CLIP, SimCLR, and DINO",
        "slug":  "contrastive-self-supervised-learning",
        "tags":  "contrastive ssl simclr moco dino clip",
        "cat":   "foundation-models",
        "desc":  "SimCLR, MoCo, BYOL, and DINO — the elegant mathematics of learning powerful representations by contrasting augmented views, without any labels.",
        "yt":    "APki_-mdBus",
        "focus": """
Instance discrimination pretext task. Data augmentation curriculum.
SimCLR: NT-Xent loss, large-batch requirement, projection head.
MoCo v1/v2/v3: momentum encoder + queue — decouples batch size from negative count.
BYOL: online + target networks, stop-gradient — no negatives needed.
DINO: self-distillation with centering + sharpening; ViT features work as k-NN classifier.
ECharts line chart: linear probe top-1 on ImageNet vs pretraining epochs
for SimCLR / MoCo-v2 / BYOL / DINO / DINOv2.
Math: NT-Xent L = -Σ_i log sim(z_i, z'_i)/τ / Σ_{j≠i} sim(z_i, z'_j)/τ.
PyTorch: SimCLR projection head + NT-Xent loss.""",
    },
    {
        "title": "Graph Neural Networks and Foundation Models for Science",
        "slug":  "graph-neural-networks-foundation",
        "tags":  "gnn graph molecular drug-discovery alphafold",
        "cat":   "applications",
        "desc":  "How GNNs and graph-aware Transformers are enabling breakthroughs in drug discovery, materials science, and protein structure prediction.",
        "yt":    "uF53jwOKkq8",
        "focus": """
MPNN, GraphSAGE, GAT (graph attention), GIN (most expressive WL-equivalent).
Graph Transformers: Graphormer (biases from graph distance + centrality), GPS.
AlphaFold 2: Evoformer as a graph-aware module over residue pairs.
Molecular property prediction: QM9 and OGB benchmarks.
Pre-training on graphs: masked node/edge prediction, contrastive.
ECharts graph type (force-directed network): caffeine molecule —
atom nodes colored by element, bond edges.
Math: h_v^{(k)} = UPDATE(h_v^{(k-1)}, AGGREGATE({h_u^{(k-1)}: u ∈ N(v)})).
PyTorch Geometric: 2-layer GAT on molecular graph.""",
    },
    {
        "title": "Knowledge Distillation: Teaching Small Models to Think Big",
        "slug":  "knowledge-distillation",
        "tags":  "distillation compression pruning quantization",
        "cat":   "efficiency",
        "desc":  "How knowledge distillation, pruning, and quantization compress state-of-the-art models into deployable systems — without sacrificing capability.",
        "yt":    "BHnTCBKCLZ8",
        "focus": """
Hinton et al. KD: soft targets + temperature scaling — why soft labels carry 'dark knowledge'.
DistilBERT, TinyBERT: intermediate layer distillation (attention maps + hidden states).
Feature-map distillation, attention transfer, relational KD.
SparseGPT structured pruning, GPTQ and AWQ post-training quantization.
ECharts bubble chart: model size (MB) vs GLUE accuracy vs inference latency (ms/token)
for BERT-large / BERT-base / DistilBERT / TinyBERT / MobileBERT.
Math: L_KD = α H(y_hard, σ(z_s)) + (1-α) τ² KL(σ(z_t/τ) || σ(z_s/τ)).
PyTorch: knowledge distillation training loop with temperature softmax.""",
    },
    {
        "title": "In-Context Learning: How LLMs Learn Without Gradient Updates",
        "slug":  "in-context-learning",
        "tags":  "icl few-shot prompting meta-learning llm",
        "cat":   "foundation-models",
        "desc":  "The mysterious emergent ability of large language models to perform new tasks from just a handful of examples in the prompt — no gradient updates required.",
        "yt":    "pVJbGzJMGbQ",
        "focus": """
ICL definition: task inference from labeled examples in context only.
What matters: label space coverage, input-label format, input distribution (not label correctness!).
Mechanistic explanation: induction heads implement ICL as implicit gradient descent.
Akyürek et al.: Transformers implement linear regression in-context.
Meta-ICL and instruction tuning (FLAN, T0) as enabling zero-shot ICL.
ECharts line chart: classification accuracy vs number of demonstrations (0–32)
for GPT-2-XL / GPT-3 / InstructGPT / GPT-4 on SST-2.
Math: Bayesian view P(y|x, C) ∝ P(C|y, x) · P(y|x); implicit prior from pretraining.
Python: systematic few-shot prompt builder with label calibration.""",
    },
]

# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a senior ML researcher and gifted science communicator.
You write technical blog posts for sadjadalikhani.github.io (Jekyll al-folio theme).

Your posts are:
• Technically rigorous yet accessible to graduate students
• Visually rich: interactive ECharts, LaTeX math, Python code, YouTube embeds
• First-principles in style — build intuition before formalism
• Impeccably structured

════════ CRITICAL FORMATTING RULES ════════

1. NO YAML front matter — the script inserts it. Start directly with prose.

2. MATH: ALWAYS use $$ ... $$ — this is the ONLY math delimiter that renders.
   NEVER use \( ... \) or \[ ... \] or single $ — they will appear as raw text.
   Inline example: the weight matrix $$\mathbf{W} \in \mathbb{R}^{d \times d}$$ is learned.
   Display math — put on its own paragraph with a BLANK LINE before AND after:

   $$
   y = \mathbf{W}x + b
   $$

3. ECHARTS: fenced block with language "echarts", valid JSON only (no // comments,
   no trailing commas). Example:
   ```echarts
   {
     "title": { "text": "My Chart" },
     "xAxis": { "data": ["A","B","C"] },
     "yAxis": {},
     "series": [{ "type": "bar", "data": [10, 20, 30] }]
   }
   ```

4. YOUTUBE embed — use exactly this pattern (replace VIDEO_ID):
   <div class="row mt-3">
     <div class="col-sm mt-3 mt-md-0">
       {% include video.liquid path="https://www.youtube.com/embed/VIDEO_ID" class="img-fluid rounded z-depth-1" %}
     </div>
   </div>
   <div class="caption">One-sentence caption here.</div>

5. IMAGES: only markdown with public Unsplash URLs:
   ![descriptive alt text](https://images.unsplash.com/photo-XXXXX?w=800&q=80)
   Do NOT use {% include figure.liquid %}.

6. BLOCKQUOTES (al-folio style):
   > Regular quote with — Attribution

   > ##### TIP
   > Key insight the reader should remember.
   {: .block-tip }

   > ##### WARNING
   > A common pitfall or misconception.
   {: .block-warning }

7. CODE: always specify the language: ```python, ```bash, ```text, etc.
8. No raw <script> tags. No <img> tags.

════════ REQUIRED STRUCTURE ════════

1. Opening hook — 2-3 vivid sentences that make the reader lean forward
2. Famous pull-quote: > "quote" \\n> — Name, Year
3. ## The Core Intuition  (analogy + accessible explanation, 200-300 words)
4. ## The Mathematics  (rigorous derivation, 250-400 words with display math)
5. YouTube embed with caption
6. ## Architecture & Implementation  (Python/PyTorch code block, 200-300 words explanation)
7. ## Benchmarks & Performance  (ECharts chart + 150-200 words analysis)
8. ## Real-World Impact & Open Problems  (150-200 words)
9. .block-tip TIP blockquote — the single most important insight
10. .block-warning WARNING blockquote — the most common mistake
11. ## Further Reading  (5 real papers: Title — Authors, Year. No URLs.)

TARGET: 1500–2000 words of prose (not counting code/math/charts).
TONE: Authoritative, enthusiastic, every sentence earns its place."""


def build_user_prompt(topic: dict) -> str:
    return (
        f"Write a complete blog post.\n\n"
        f"**Title:** {topic['title']}\n"
        f"**One-line description:** {topic['desc']}\n\n"
        f"**Required technical content:**\n{topic['focus']}\n\n"
        f"**YouTube embed:** use ID `{topic['yt']}` in the embed template. "
        f"Place it after the Mathematics section.\n\n"
        "Write the full post body now. No front matter.\n"
        "ECharts data must be realistic — use plausible benchmark numbers.\n"
        "Python code must be Python 3.11+, PyTorch 2.x, idiomatic and runnable.\n"
        "LaTeX: use \\\\mathbf, \\\\mathbb, \\\\text where appropriate."
    )


# ── OpenAI API ─────────────────────────────────────────────────────────────────
def call_claude(user_prompt: str) -> str:
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "content-type": "application/json",
        },
        json={
            "model": "gpt-4o",
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        },
        timeout=180,
    )
    if resp.status_code != 200:
        sys.exit(f"OpenAI API error {resp.status_code}:\n{resp.text[:600]}")
    return resp.json()["choices"][0]["message"]["content"]


# ── State ──────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"next_index": 0, "published": []}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Front matter ───────────────────────────────────────────────────────────────
def build_front_matter(topic: dict, today: str) -> str:
    return (
        "---\n"
        "layout: post\n"
        f'title: "{topic["title"]}"\n'
        f"date: {today} 09:00:00\n"
        f'description: "{topic["desc"]}"\n'
        f"tags: {topic['tags']}\n"
        f"categories: {topic['cat']}\n"
        "chart:\n"
        "  echarts: true\n"
        "related_posts: true\n"
        "toc:\n"
        "  sidebar: left\n"
        "---\n"
    )


def sanitize(content: str) -> str:
    """Strip any accidental front matter the model may emit."""
    content = content.strip()
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()
    return content


def fix_math(content: str) -> str:
    """
    Convert all LaTeX delimiters to the $$ form that al-folio / MathJax expects.
    The model sometimes emits \\( ... \\) for inline and \\[ ... \\] for display
    despite instructions — this normalises both.
    """
    import re
    # Inline: \( ... \)  →  $$ ... $$
    content = re.sub(r'\\\(\s*(.*?)\s*\\\)', r'$$\1$$', content, flags=re.DOTALL)
    # Display: \[ ... \]  →  \n$$\n...\n$$\n  (with surrounding blank lines)
    def display_sub(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f"\n\n$$\n{inner}\n$$\n\n"
    content = re.sub(r'\\\[\s*(.*?)\s*\\\]', display_sub, content, flags=re.DOTALL)
    return content


def fix_youtube(content: str, topic_title: str) -> str:
    """
    Validate every YouTube embed ID against the oEmbed API (no auth needed).
    If a video is unavailable, replace the embed with a YouTube search button.
    """
    import re
    pattern = re.compile(
        r'({% include video\.liquid path="https://www\.youtube\.com/embed/([^"]+)"[^%]*%})'
    )
    search_query = topic_title.replace(" ", "+")
    fallback_link = (
        f'<a href="https://www.youtube.com/results?search_query={search_query}" '
        f'target="_blank" class="btn btn-sm z-depth-0" role="button" '
        f'style="background:#ff0000;color:#fff;">▶ Watch on YouTube</a>'
    )

    def check_and_replace(m: re.Match) -> str:
        full_tag, video_id = m.group(1), m.group(2)
        try:
            r = requests.get(
                "https://www.youtube.com/oembed",
                params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
                timeout=8,
            )
            if r.status_code == 200:
                return full_tag  # video exists — keep the embed
        except Exception:
            pass
        print(f"  ⚠ YouTube ID '{video_id}' unavailable — replacing with search link.")
        return fallback_link

    return pattern.sub(check_and_replace, content)


def postprocess(content: str, topic: dict) -> str:
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = fix_math(content)
    content = fix_youtube(content, topic["title"])
    return content


# ── Git ────────────────────────────────────────────────────────────────────────
def git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(
        ["git"] + args, cwd=str(cwd), capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{r.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def commit_and_push(post_path: Path, config_path: Path, topic: dict) -> None:
    """Stage the new post + updated state, commit, and push."""
    print("  → git add …")
    git(["add", str(post_path.relative_to(SITE_DIR))], SITE_DIR)
    git(["add", str(STATE_FILE.relative_to(SITE_DIR))], SITE_DIR)
    print("  → git commit …")
    git(["commit", "-m", f"blog: {topic['title']}"], SITE_DIR)
    print("  → git push …")
    git(["push"], SITE_DIR)
    print("  ✓ Pushed to GitHub Pages.")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    dry_run = "--dry-run" in sys.argv
    force   = "--force"   in sys.argv

    state = load_state()
    idx   = state["next_index"] % len(TOPICS)
    topic = TOPICS[idx]
    today = date.today().strftime("%Y-%m-%d")

    outpath = SITE_DIR / "_posts" / f"{today}-{topic['slug']}.md"

    if outpath.exists() and not force:
        print(f"Post already exists: {outpath.name}  (pass --force to overwrite)")
        sys.exit(0)

    print(f"[{today}] Topic {idx + 1}/{len(TOPICS)}: {topic['title']}")
    print("  → Calling OpenAI gpt-4o …")
    body = postprocess(sanitize(call_claude(build_user_prompt(topic))), topic)

    full = build_front_matter(topic, today) + "\n" + body + "\n"
    print(f"  → Writing {outpath.name}  ({len(full):,} chars) …")
    outpath.write_text(full, encoding="utf-8")

    # Advance state only after successfully writing
    state["next_index"] = (idx + 1) % len(TOPICS)
    state["published"].append({
        "date": today, "slug": topic["slug"], "title": topic["title"]
    })
    save_state(state)

    if dry_run:
        print("  [dry-run] Skipping git commit / push.")
    else:
        commit_and_push(outpath, SITE_DIR / "_config.yml", topic)

    print("Done!")


if __name__ == "__main__":
    main()
