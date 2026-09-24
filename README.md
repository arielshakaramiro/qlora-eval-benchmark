# QLoRA Evaluation Benchmark

A small, controlled benchmark for what QLoRA fine-tuning actually learns and how fast — using two synthetic customer-service domains, held-out prompts, and an epoch sweep on a 0.5B model that's cheap enough to run three times over.

Everything below comes from an actual Google Colab run. No number is estimated.

## What this does

1. Computes LoRA parameter counts and 4-bit memory estimates analytically for three model sizes (Qwen2.5-0.5B, LLaMA-7B, Llama-3.1-8B), then verifies the estimate against a real measurement.
2. Builds two fictional customer-service domains (**GadaiKita**, a pawnshop; **TeknoMart**, an electronics retailer), each with 10 facts × 6 phrasings = 60 training examples.
3. Splits evaluation into three sets per domain: `train_exact` (identical to training data, tests memorization), `paraphrase` (same meaning, new wording, tests generalization), and `heldout` (facts never seen in training).
4. Automatically checks each answer for format compliance, correct brand name, presence of key facts, and repetition — no manual reading required to get a score.
5. Trains each domain at 0, 3, and 15 epochs and compares the resulting metrics.
6. Loads two LoRA adapters onto one shared base model and switches between them with `set_adapter`.

## Results

### Format is learned long before facts

At epoch 3, both domains already answer with the right greeting/closing template on `train_exact` prompts — but get 0% of the facts right, even on questions copied verbatim from training data:

| Domain | Epoch | `format_pct` (train_exact) | `fact_pct` (train_exact) |
|---|---|---|---|
| GadaiKita | 3 | 70% | 0% |
| GadaiKita | 15 | 100% | 100% |
| TeknoMart | 3 | 70% | 0% |
| TeknoMart | 15 | 100% | 100% |

Facts generalize far less than the template does — at epoch 15, `fact_pct` on *paraphrased* questions is only 40% (GadaiKita) and 50% (TeknoMart), despite 100% on the literal training questions:

![Format learned faster than facts](images/format_vs_fact.png)

### Brand-name drift is real, and epoch-dependent

At epoch 3, GadaiKita already gets its own name right 100% of the time. TeknoMart doesn't — some answers say "TeknoSupport" or "TeknoMart Support" instead:

| Domain | Epoch | `brand_pct` (paraphrase) | `brand_pct` (heldout) |
|---|---|---|---|
| GadaiKita | 3 | 100% | 100% |
| TeknoMart | 3 | 60% | 75% |
| GadaiKita | 15 | 100% | 100% |
| TeknoMart | 15 | 100% | 100% |

By epoch 15 the drift is gone for both domains:

![Brand drift by epoch](images/brand_drift.png)

### 4-bit memory: measured, not assumed

| Model | FP16 size | 4-bit estimate | Measured (Qwen2.5-0.5B only) | Ratio |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.99 GB | 0.46 GB | **0.45 GB** | 2.19x |
| LLaMA-7B (original) | 13.48 GB | 3.87 GB | — | 3.49x (est.) |
| Llama-3.1-8B | 16.06 GB | 5.70 GB | — | 2.82x (est.) |

The measured ratio (2.19x) is consistent with the estimate (2.16x) and, together with the [companion fine-tuning repo](https://github.com/arielshakaramiro/llama3-qlora-finetuning)'s direct measurement on an 8B model, does not support a flat "4x smaller" rule of thumb for 4-bit quantization.

## An automated leak guard, and a non-monotonic finding it helped surface

Every `heldout` and `paraphrase` question is checked programmatically against the training set before any training runs — the notebook raises an `AssertionError` if it finds overlap, rather than silently producing a biased score. An earlier version of this benchmark had two `heldout` questions that named the brand directly in the question text (e.g. *"Does GadaiKita have a branch in Surabaya?"*), which let the untrained baseline model score 50% on `brand_pct` just by echoing the question. Those questions were rewritten to remove the brand name, and the guard now confirms zero leakage on every run.

With that fixed, one more finding held up on a second independent run: TeknoMart's `format_pct` on `heldout` questions actually **drops** from 75% (epoch 3) to 50% (epoch 15), even as `brand_pct` on the same set rises to 100%. Training loss at epoch 15 is very low (0.33, down from 1.55 at epoch 3) — a sign of overfitting on just 60 examples. One epoch-15 answer includes a nonsensical Indonesian phrase ("...berlaku hanya di masa **abat-kata**...") — not a wrong fact, just incoherent. Heavier training locked in the brand name perfectly while destabilizing sentence-level coherence on prompts the model never saw.

## How to run

1. Open `notebooks/qlora_hf_stack_evaluation.ipynb` in Google Colab.
2. Set the runtime to a GPU (T4 is enough — the model is 0.5B).
3. Run all cells top to bottom. A Hugging Face token is optional.
4. Adapters are saved to Google Drive under `AI-Engineer/LLM/qlora-customer-service/`.

Approximate total runtime: 10–15 minutes.

## Repo structure

```
.
├── notebooks/
│   └── qlora_hf_stack_evaluation.ipynb
├── images/
│   ├── format_vs_fact.png
│   └── brand_drift.png
├── LICENSE
├── README.md
└── README.id.md
```

## Limitations

- This is a small-scale, single-seed experiment (10 facts per domain, one run per epoch setting) — differences of a few percentage points should not be read as stable.
- The automated checks (format, brand, fact keyword presence) are keyword-based; an answer can contain the right keywords in the wrong context and still score as correct.
- Both domains are fictional and built specifically for this benchmark.

## License

MIT — see [LICENSE](LICENSE).
