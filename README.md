# Search-R1-Improve: Stable GRPO Training for Search Agents

<p align="center">
  <strong>English</strong> |
  <a href="README_CN.md">简体中文</a>
</p>

This repository extends [Search-R1](https://github.com/PeterGriffinJin/Search-R1), a reinforcement learning framework for training language models to interleave reasoning and search-engine calls.

The main improvement is an action-gated likelihood-decline regularizer inspired by *On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement*. The regularizer protects useful trajectories from unintended likelihood reduction during GRPO updates. This repository also includes larger context budgets, longer search trajectories, loss clipping, multi-dataset preprocessing, and training-log visualization.

> This is an independent implementation built on Search-R1. It is not the official implementation of the LLD/LLDS paper.

## Highlights

- Action-level likelihood-decline regularization for GRPO.
- Protection restricted to responses with non-negative advantages.
- Token-level penalties applied only where likelihood decreases.
- Optional answer-span weighting for multi-step search experiments.
- Support for NQ and NQ+HotpotQA training.
- Evaluation grouped by dataset through `val/test_score/<dataset>`.
- Utilities for plotting training metrics and validation curves.
- Increased context, observation, response, and search-turn budgets.

## Method

### Motivation

In tool-integrated GRPO, an incorrect final answer can assign a negative update to an otherwise useful search action. As training continues, the likelihood of useful actions may decrease even while the training reward initially improves. This can produce low-confidence trajectories, higher entropy, unstable gradients, and eventual training collapse.

For each generated token, this project measures the likelihood decline:

```text
delta = log pi_old(token) - log pi_current(token)
```

A positive `delta` means that the current update has reduced the token likelihood.

### Likelihood-decline regularization

The regularizer applies three filters:

1. Only trajectories with `advantage >= 0` are protected.
2. An action is protected only when its total likelihood decreases.
3. Within an active action, only tokens with positive likelihood decline are penalized.

Conceptually:

```text
L_LLD =
    mean(
        1[action likelihood decreased]
        * 1[advantage >= 0]
        * max(0, log pi_old - log pi_current)
    )

L_total = L_GRPO + lld_coef * L_LLD
```

Tool feedback tokens are excluded through Search-R1's state mask. Consecutive trainable spans separated by masked tool observations are treated as action chunks.

### Configuration

The custom actor options are:

```yaml
actor_rollout_ref:
  actor:
    lld_coef: 0.0
    lld_action_gate: false
    lld_mask_answer: false
    lld_answer_coef: 0.0
    lld_adaptive_answer_coef: false
    lld_clip_nan_grad: false
```

The main experiments use:

```text
lld_coef = 0.2
lld_action_gate = true
lld_mask_answer = false
```

Setting `lld_coef=0.0` recovers the GRPO baseline for controlled comparisons.

## Experimental Results

We evaluated Qwen2.5-3B-Instruct on four multi-hop QA benchmarks. Scores follow the Search-R1 rule-based evaluation protocol.

| Method | HotpotQA | 2Wiki | Musique | Bamboogle | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Search-R1 | 0.331 | 0.310 | 0.124 | 0.232 | 0.249 |
| **Search-R1-Improve** | **0.400** | **0.384** | **0.159** | **0.320** | **0.316** |

Absolute improvements:

| Dataset | Improvement |
| --- | ---: |
| HotpotQA | +0.069 |
| 2Wiki | +0.074 |
| Musique | +0.035 |
| Bamboogle | +0.088 |
| Average | +0.067 |

The average multi-hop score improves from `0.249` to `0.316`, corresponding to an approximate relative improvement of `26.7%`.

## Repository Structure

```text
Search-R1/
|-- search_r1/                 # Multi-turn generation and retrieval interaction
|-- verl/                      # PPO/GRPO training framework
|   |-- trainer/ppo/           # Advantage and policy-loss implementations
|   `-- workers/actor/         # Actor update and LLD regularization
|-- scripts/data_process/      # QA dataset preprocessing
|-- public/                    # README assets
|-- train_grpo.sh              # GRPO training entry point
|-- train_ppo.sh               # PPO training entry point
|-- retrieval_launch.sh        # Local retrieval server
|-- infer.py                   # Interactive inference
|-- plot_metrics.py            # Training dashboard generation
`-- plot_val_nq.py             # Validation-score plotting
```

## Installation

### Training environment

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb matplotlib
```

### Retriever environment

Using a separate environment for the retriever is recommended:

```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install transformers datasets pyserini uvicorn fastapi
```

## Data Preparation

### NQ-only

```bash
python scripts/data_process/nq_search.py \
  --local_dir ./data/nq_search
```

This generates:

```text
data/nq_search/train.parquet
data/nq_search/test.parquet
```

### NQ + HotpotQA training

The extended preprocessing script supports separate training and evaluation dataset lists:

```bash
python scripts/data_process/nq_search.py \
  --local_dir ./data/nq_hotpotqa_train \
  --train_data_sources nq,hotpotqa \
  --test_data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle \
  --cache_dir ./tmp
```

Training data:

```text
NQ + HotpotQA
```

Evaluation data:

```text
NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle
```

To reduce validation cost during training, set:

```bash
data.val_data_num=704
```

The validation subset is sampled with a fixed random seed. For final reporting, evaluate the selected checkpoint on the complete test set.

## Retriever Setup

Download the Search-R1 Wikipedia corpus and E5 index:

```bash
save_path=/path/to/retrieval_data
python scripts/download.py --save_path "$save_path"
cat "$save_path"/part_* > "$save_path"/e5_Flat.index
gzip -d "$save_path"/wiki-18.jsonl.gz
```

Update `retrieval_launch.sh` with the local corpus and index paths, then launch:

```bash
conda activate retriever
bash retrieval_launch.sh
```

Test the endpoint before training:

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["What is machine learning?"],"topk":3,"return_scores":true}'
```

The response should contain a JSON `result` field. An HTML `502 Bad Gateway` response indicates that the retriever or proxy is unavailable.

## Training

Before running any script, replace every quoted `<path-to-...>` placeholder with the corresponding local path or Hugging Face model ID.

The recommended starting configuration for Qwen2.5-3B is:

| Parameter | Value |
| --- | ---: |
| `max_prompt_length` | 16384 |
| `max_start_length` | 1024 |
| `max_response_length` | 1024 |
| `max_obs_length` | 2048 |
| `max_turns` | 4 |
| `n_agent` | 5 |
| `lld_coef` | 0.2 |
| `lld_action_gate` | true |
| `lld_mask_answer` | false |

Enable the method in `train_grpo.sh`:

```bash
export LLD_COEF=0.2
export LLD_ACTION_GATE=true
export LLD_MASK_ANSWER=false
export LLD_ANSWER_COEF=0.0
export LLD_ADAPTIVE_ANSWER_COEF=false
export LLD_CLIP_NAN_GRAD=true
```

Pass the values to Hydra:

```bash
actor_rollout_ref.actor.lld_coef=$LLD_COEF \
actor_rollout_ref.actor.lld_action_gate=$LLD_ACTION_GATE \
actor_rollout_ref.actor.lld_mask_answer=$LLD_MASK_ANSWER \
actor_rollout_ref.actor.lld_answer_coef=$LLD_ANSWER_COEF \
actor_rollout_ref.actor.lld_adaptive_answer_coef=$LLD_ADAPTIVE_ANSWER_COEF \
actor_rollout_ref.actor.lld_clip_nan_grad=$LLD_CLIP_NAN_GRAD \
```

Start GRPO training:

```bash
conda activate searchr1
bash train_grpo.sh
```

For a short smoke test, temporarily use:

```text
trainer.total_training_steps=2
trainer.test_freq=100000
trainer.save_freq=100000
```

## Training Metrics

The implementation logs the custom metrics:

```text
actor/lld_loss
actor/lld_active_ratio
actor/lld_coef
```

Interpretation:

- `actor/lld_loss`: magnitude of the likelihood-decline penalty.
- `actor/lld_active_ratio`: fraction of valid tokens covered by active action gates.
- `actor/lld_coef`: configured regularization coefficient.

Useful stability metrics include:

```text
actor/entropy_loss
actor/grad_norm
actor/pg_loss
actor/kl_loss
critic/score/mean
env/number_of_valid_search
```

Validation is reported separately for each dataset:

```text
val/test_score/nq
val/test_score/triviaqa
val/test_score/popqa
val/test_score/hotpotqa
val/test_score/2wikimultihopqa
val/test_score/musique
val/test_score/bamboogle
```

## Visualization

Generate the full training dashboard:

```bash
python plot_metrics.py path/to/training.log
```

Plot the NQ validation curve:

```bash
python plot_val_nq.py path/to/training.log \
  --metric val/test_score/nq \
  --ma 3
```

Plot another benchmark:

```bash
python plot_val_nq.py path/to/training.log \
  --metric val/test_score/hotpotqa \
  --out plots/hotpotqa.png
```

## Inference

Launch the retriever first, then run:

```bash
conda activate searchr1
python infer.py
```

## Notes on Reproduction

- Use the same base model, retriever, corpus, rollout count, and validation split for fair comparisons.
- Compare against a baseline with `lld_coef=0.0` while keeping all other settings unchanged.
- The regularizer may reduce performance when the baseline is already stable or when its coefficient is too large.
- Recommended coefficient ablations are `0.05`, `0.1`, and `0.2`.
- Qwen2.5-3B may benefit from reduced answer-token regularization when it collapses to single-search behavior.
- Select checkpoints using validation scores rather than training reward alone.

## Acknowledgements

This project is built on:

- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [veRL](https://github.com/volcengine/verl)
- [RAGEN](https://github.com/ZihanWang314/RAGEN)

The likelihood-decline regularization is inspired by:

- *On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement*

Please follow the licenses and citation requirements of the upstream projects.

## Citation

If you use the original Search-R1 framework, cite:

```bibtex
@article{jin2025search,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv preprint arXiv:2505.15117},
  year={2025}
}
```

For the LLD analysis and official LLDS method, cite the corresponding paper:

```bibtex
@article{deng2026lazy,
  title={On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement},
  author={Deng, Wenlong and Li, Yushu and Gong, Boying and Ren, Yi and Thrampoulidis, Christos and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2512.04220},
  year={2026}
}
```

## License

This repository follows the upstream Search-R1 licensing terms. See [LICENSE](LICENSE) for details.
