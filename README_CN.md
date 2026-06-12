# Search-R1-Improve：面向搜索智能体的稳定 GRPO 训练

<p align="center">
  <a href="README.md">English</a> |
  <strong>简体中文</strong>
</p>

本仓库扩展自 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)。Search-R1 是一个强化学习框架，用于训练语言模型交替执行推理与搜索引擎调用。

本项目的主要改进是一种 action 级门控的似然下降正则方法，其思想受到论文 *On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement* 启发。该正则项用于防止 GRPO 更新过程中有用轨迹的似然被意外降低。此外，本仓库还加入了更大的上下文预算、更长的搜索轨迹、loss 裁剪、多数据集预处理以及训练日志可视化工具。

> 本项目是基于 Search-R1 的独立实现

## 主要特性

- 面向 GRPO 的 action 级似然下降正则。
- 只保护 advantage 非负的 response。
- 只惩罚似然实际下降的 token。
- 支持对最终答案 span 设置独立权重，用于多步搜索实验。
- 按数据集记录 `val/test_score/<dataset>` 验证指标。
- 提供训练指标和验证曲线可视化工具。
- 扩大上下文、工具反馈、response 和搜索轮数预算。

## 方法

### 动机

在工具集成的 GRPO 训练中，一个错误的最终答案可能会让前面本来有效的搜索 action 接收到负向更新。随着训练继续，即使训练 reward 在前期仍然上升，有用 action 的似然也可能持续下降。这可能进一步产生低置信度轨迹、熵增大、梯度不稳定，最终导致训练崩溃。

对于每个生成 token，本项目计算它的似然下降量：

```text
delta = log pi_old(token) - log pi_current(token)
```

当 `delta` 为正时，说明当前策略更新降低了该 token 的似然。

### 似然下降正则

正则项包含三层筛选：

1. 只保护满足 `advantage >= 0` 的轨迹。
2. 只有当一个 action 的总似然下降时，才激活该 action 的保护。
3. 在已激活的 action 内，只惩罚似然下降的 token。

其概念形式为：

```text
L_LLD =
    mean(
        1[action likelihood decreased]
        * 1[advantage >= 0]
        * max(0, log pi_old - log pi_current)
    )

L_total = L_GRPO + lld_coef * L_LLD
```

Search-R1 的 state mask 会排除工具反馈 token。被工具反馈隔开的连续可训练 token span 被视为独立的 action chunk。

### 配置项

自定义 actor 配置如下：

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

主要实验使用：

```text
lld_coef = 0.2
lld_action_gate = true
lld_mask_answer = false
```

将 `lld_coef=0.0` 即可关闭该方法，恢复为 GRPO baseline，便于进行控制变量对比。

## 实验结果

我们使用 Qwen2.5-3B-Instruct，在四个多跳问答数据集上进行了评估。分数采用 Search-R1 的规则奖励评估方式。

| 方法 | HotpotQA | 2Wiki | Musique | Bamboogle | 平均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Search-R1 | 0.331 | 0.310 | 0.124 | 0.232 | 0.249 |
| **Search-R1-Improve** | **0.400** | **0.384** | **0.159** | **0.320** | **0.316** |

## 项目结构

```text
Search-R1/
|-- search_r1/                 # 多轮生成与检索交互
|-- verl/                      # PPO/GRPO 训练框架
|   |-- trainer/ppo/           # Advantage 与 policy loss 实现
|   `-- workers/actor/         # Actor 更新与 LLD 正则
|-- scripts/data_process/      # QA 数据预处理
|-- public/                    # README 静态资源
|-- train_grpo.sh              # GRPO 训练入口
|-- train_ppo.sh               # PPO 训练入口
|-- retrieval_launch.sh        # 本地检索服务
|-- infer.py                   # 交互式推理
|-- plot_metrics.py            # 训练指标面板
`-- plot_val_nq.py             # 验证分数曲线
```

## 环境安装

### 训练环境

```bash
conda create -n searchr1 python=3.9
conda activate searchr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb matplotlib
```

### 检索环境

建议为检索服务创建单独的环境：

```bash
conda create -n retriever python=3.10
conda activate retriever

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

pip install transformers datasets pyserini uvicorn fastapi
```

## 数据准备

### 仅使用 NQ

```bash
python scripts/data_process/nq_search.py \
  --local_dir ./data/nq_search
```

生成文件：

```text
data/nq_search/train.parquet
data/nq_search/test.parquet
```

### 使用 NQ + HotpotQA 训练

扩展后的预处理脚本支持分别指定训练集和测试集：

```bash
python scripts/data_process/nq_search.py \
  --local_dir ./data/nq_hotpotqa_train \
  --train_data_sources nq,hotpotqa \
  --test_data_sources nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle \
  --cache_dir ./tmp
```

训练数据：

```text
NQ + HotpotQA
```

评估数据：

```text
NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、Musique、Bamboogle
```

为了降低训练过程中的验证开销，可以设置：

```bash
data.val_data_num=704
```

验证子集会使用固定随机种子抽取。正式报告结果时，建议使用选定的 checkpoint 在完整测试集上重新评估。

## 检索服务

下载 Search-R1 使用的 Wikipedia 语料和 E5 索引：

```bash
save_path=/path/to/retrieval_data
python scripts/download.py --save_path "$save_path"
cat "$save_path"/part_* > "$save_path"/e5_Flat.index
gzip -d "$save_path"/wiki-18.jsonl.gz
```

修改 `retrieval_launch.sh` 中的语料和索引路径，然后启动：

```bash
conda activate retriever
bash retrieval_launch.sh
```

训练前测试检索接口：

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries":["What is machine learning?"],"topk":3,"return_scores":true}'
```

正常响应中应包含 JSON `result` 字段。如果返回 HTML 格式的 `502 Bad Gateway`，说明检索服务或代理不可用。

## 训练

运行脚本前，请将所有带引号的 `<path-to-...>` 占位符替换为对应的本地路径或 Hugging Face 模型 ID。

Qwen2.5-3B 推荐的起始配置如下：

| 参数 | 数值 |
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

在 `train_grpo.sh` 中启用该方法：

```bash
export LLD_COEF=0.2
export LLD_ACTION_GATE=true
export LLD_MASK_ANSWER=false
export LLD_ANSWER_COEF=0.0
export LLD_ADAPTIVE_ANSWER_COEF=false
export LLD_CLIP_NAN_GRAD=true
```

将这些值传给 Hydra：

```bash
actor_rollout_ref.actor.lld_coef=$LLD_COEF \
actor_rollout_ref.actor.lld_action_gate=$LLD_ACTION_GATE \
actor_rollout_ref.actor.lld_mask_answer=$LLD_MASK_ANSWER \
actor_rollout_ref.actor.lld_answer_coef=$LLD_ANSWER_COEF \
actor_rollout_ref.actor.lld_adaptive_answer_coef=$LLD_ADAPTIVE_ANSWER_COEF \
actor_rollout_ref.actor.lld_clip_nan_grad=$LLD_CLIP_NAN_GRAD \
```

启动 GRPO 训练：

```bash
conda activate searchr1
bash train_grpo.sh
```

进行短流程测试时，可以临时设置：

```text
trainer.total_training_steps=2
trainer.test_freq=100000
trainer.save_freq=100000
```

## 训练指标

实现中会记录以下自定义指标：

```text
actor/lld_loss
actor/lld_active_ratio
actor/lld_coef
```

指标含义：

- `actor/lld_loss`：似然下降正则项的大小。
- `actor/lld_active_ratio`：被 action gate 激活的有效 token 比例。
- `actor/lld_coef`：配置的正则权重。

建议同时观察以下训练稳定性指标：

```text
actor/entropy_loss
actor/grad_norm
actor/pg_loss
actor/kl_loss
critic/score/mean
env/number_of_valid_search
```

验证指标会按照数据集分别记录：

```text
val/test_score/nq
val/test_score/triviaqa
val/test_score/popqa
val/test_score/hotpotqa
val/test_score/2wikimultihopqa
val/test_score/musique
val/test_score/bamboogle
```

## 可视化

生成完整训练指标面板：

```bash
python plot_metrics.py path/to/training.log
```

绘制 NQ 验证曲线：

```bash
python plot_val_nq.py path/to/training.log \
  --metric val/test_score/nq \
  --ma 3
```

绘制其他数据集：

```bash
python plot_val_nq.py path/to/training.log \
  --metric val/test_score/hotpotqa \
  --out plots/hotpotqa.png
```

## 推理

先启动检索服务，然后运行：

```bash
conda activate searchr1
python infer.py
```

## 复现说明

- 公平对比时应保持基础模型、检索器、语料、rollout 数量和验证集一致。
- GRPO baseline 应设置 `lld_coef=0.0`，其他配置保持不变。
- 当 baseline 本身已经稳定，或正则权重过大时，该方法可能降低性能。
- 建议测试 `0.05`、`0.1` 和 `0.2` 三个正则权重。
- 当 Qwen2.5-3B 退化为单次搜索时，可以尝试降低最终答案 token 的保护权重。
- 应根据验证集分数选择 checkpoint，不要只观察训练 reward。

## 致谢

本项目基于：

- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)
- [veRL](https://github.com/volcengine/verl)
- [RAGEN](https://github.com/ZihanWang314/RAGEN)

似然下降正则方法受到以下工作启发：

- *On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement*

使用本项目时，请遵循上游项目的开源协议和引用要求。

## 引用

使用原始 Search-R1 框架时，请引用：

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

对于 LLD 分析与官方 LLDS 方法，请引用：

```bibtex
@article{deng2026lazy,
  title={On Group Relative Policy Optimization Collapse in Agent Search: The Lazy Likelihood-Displacement},
  author={Deng, Wenlong and Li, Yushu and Gong, Boying and Ren, Yi and Thrampoulidis, Christos and Li, Xiaoxiao},
  journal={arXiv preprint arXiv:2512.04220},
  year={2026}
}
```

## 开源协议

本仓库遵循上游 Search-R1 的开源协议，详情请参阅 [LICENSE](LICENSE)。
