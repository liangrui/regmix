# 🧬 RegMix：把数据混合当作回归问题来优化语言模型预训练

欢迎来到 [RegMix](https://huggingface.co/papers/2407.01492) 的官方仓库。RegMix 是一种用于优化大型语言模型（LLM）预训练数据混合的新方法！
欢迎加入我们的 [Discord](https://discord.gg/y3hpRaXGaU) 参与更多讨论！

## 🌟 什么是 RegMix？

RegMix 是一种新颖的方法，它把**数据混合选择**当成一个**回归任务**来求解。通过在多样化的数据混合配比上训练小型"代理（proxy）"模型并分析它们的表现，RegMix 训练出一个回归模型，用来预测大规模 LLM 训练时的最优数据混合配比。

![RegMix 方法示意图](misc/method_figure.png)

## 🚀 RegMix 是如何工作的？

RegMix 通过四步流程来优化 LLM 训练：

1. **生成配置（Generate Configs）**：创建各种不同的数据混合配置。
2. **训练小模型（Train Small Models）**：用这些配置去训练小型"代理"模型。
3. **拟合回归模型（Fit Regression Model）**：分析这些模型的表现（例如在 Pile-CC 上的验证损失），来构建一个可预测的回归模型。
4. **训练大模型（Train Large Model）**：用预测出的最优混合配比来训练大规模 LLM。

## 🧰 仓库里有什么？

本仓库主要分为四个组成部分：

1. [**mixture_config**](mixture_config)：用于合成与可视化数据混合配置的工具。
   - 生成多样化的数据混合配置。
   - 可视化生成的混合配比。

2. [**regression_fitting**](regression_fitting)：RegMix 的核心。
   - 使用小型模型的表现数据来拟合回归模型，默认使用 Pile-CC 数据集上的验证损失。
   - 模拟并预测大规模训练所需的最优数据混合配比。

3. [**model_training**](model_training)：复用 [TinyLlama](https://github.com/jzhang38/TinyLlama) 进行模型训练。
   - 训练小型（1M 参数）代理模型。
   - 使用预测出的最优混合配比，扩展到大型（1B+ 参数）语言模型。

4. [**evaluation**](evaluation)：（开发中）复现本仓库的评估结果。

## 🛠 如何在你的数据集上应用 RegMix

想在你的自有语言模型训练中发挥 RegMix 的威力吗？按照以下步骤，把 RegMix 应用到你的专属数据集：

### 1. 准备你的数据

- **组织你的数据集**：把数据按不同的类别或领域拆分，每个领域使用唯一的前缀（prefix）。
- **格式要求**：确保数据使用兼容的格式（例如 JSON lines，每行都是一个合法的 JSON 对象，包含 `text` 字段）。

一个示例的数据集目录结构如下：

```
├── train
│   ├── domain1-0.jsonl
│   ├── domain1-1.jsonl
│   ├── ...
│   ├── domain2-0.jsonl
│   ├── domain2-1.jsonl
│   ├── ...
├── valid
│   ├── domain1-0.jsonl
│   ├── domain2-0.jsonl
```

> [!CAUTION]
> 我们使用基于前缀的匹配来识别属于同一领域的文件。因此请务必确保每个领域的前缀是唯一的。同时，请在前缀之后使用 `-` 连接符，以避免一个前缀成为另一个前缀的子串。我们的[训练代码](https://github.com/sail-sg/regmix/blob/e0c0357a312dbb3d62e3aa58a13cf42dd3ef42ee/model_training/pretrain/tinyllama.py#L464)默认使用 `-` 来确保加载到正确的领域文件！
> 请尽量避免让每个 jsonl 文件过大，否则可能导致预处理阶段耗时过长。

你还可以参考 [regmix-data-sample](https://huggingface.co/datasets/sail/regmix-data-sample) 数据样本。

### 2. 生成混合配置

使用 `mixture_config/` 目录下的工具：
- 创建一系列数据混合配置。
- 可视化这些混合配比，了解其组成。

在生成配置之前，请确保你已经修改了 `synthesize_mixture.py` 中的 `get_token_distribution` 函数，使其匹配你的数据集领域名称与 token 分布。

例如，假设你有三个领域：`domain1`、`domain2`、`domain3`，并且你在 `model_training/preprocess/run_preprocess.sh` 中定义的 `DATASET_SHORT_NAME` 为 `your_dataset`。

为了适配这些领域，你应该按如下方式修改 `get_token_distribution` 函数：

```python
def get_token_distribution():
    # 本例中为每个领域使用相同的分布。
    # 如果某些领域可用的 token 明显更多，请调整这些数值。
    train = {
        "train_your_dataset_domain1": 0.33,
        "train_your_dataset_domain2": 0.33,
        "train_your_dataset_domain3": 0.34,
    }
    # 如果无需用于 1M 模型训练，验证集可以省略
    valid = {
        "valid_your_dataset_domain1": 1.0,
        "valid_your_dataset_domain2": 1.0,
        "valid_your_dataset_domain3": 1.0,
    }
    return {"train": train, "valid": valid}
```

接下来，生成混合配置：

```bash
python synthesize_mixture.py --num_configs 512 --output_folder /path/to/configs
```

> [!TIP]
> 配置的数量通常对回归模型的准确度影响最大。在我们的实验中，17 个领域共使用了 512 个配置。如果你处理的领域数量较少，可以用更少的配置达到相近的结果。不过，如果你有额外的计算资源，也可以考虑训练更多的代理模型来提升回归模型的准确度——当领域数量较多时，这种做法尤其有益。
> 你也可以先从少量配置开始，看看回归模型能否很好地外推到未见过的数据混合。

最后，可视化生成的混合配比，了解其组成：

```bash
python visualize_mixture.py --config_folder /path/to/configs
```

### 3. 训练代理模型

使用 `model_training/` 下的脚本，在你生成的混合配比上训练小型"代理"模型。记得要修改 `pretrain_tinyllama_1m.sh` 脚本中的配置文件夹路径，使其指向你生成的配置。

```bash
cd model_training
for i in {1..512}; do
    ./pretrain_tinyllama_1m.sh $i
done
```

### 4. 拟合回归模型

有了代理模型的结果，使用收集（collect）脚本准备用于回归拟合的数据。第一步，把混合配置整理成一个 CSV 文件：

```bash
python collect_mixture_data.py --write_file_path train_mixture_1m_your_dataset.csv --config_folder /path/to/configs
```

第二步，收集代理模型的目标性能数据。默认情况下，我们使用 Pile-CC 验证损失作为目标，它是通过 `wandb` API 从 wandb 中收集的。

```bash
python collect_loss_data.py --write_file_path train_loss_1m_your_dataset.csv
```

最后，使用收集到的数据拟合回归模型，并按照 `regression_fitting/regression.ipynb` 中的说明预测最优混合配比。

### 5. 训练你的大规模 LLM

你可以把最终预测出的最优混合配比保存为一个 yaml 文件 `optimal_mixture.yaml`，其格式与 [mixture_config/config_1b](mixture_config/config_1b) 下的配置类似。最优混合的一个示例如下：

```yaml
train:
  train_the_pile_arxiv: 0.0012046169821426883
  train_the_pile_freelaw: 0.001454510048554701
  train_the_pile_nih_exporter: 0.001231640306882902
  train_the_pile_pubmed_central: 0.003108561825532002
  train_the_pile_wikipedia_en: 0.01593264140324679
  train_the_pile_dm_mathematics: 0.00031106907908634156
  train_the_pile_github: 0.00022861228152440253
  train_the_pile_philpapers: 1.329107360676338e-05
  train_the_pile_stackexchange: 0.00029547405933203174
  train_the_pile_enron_emails: 0.0016691646199353991
  train_the_pile_gutenberg_pg_19: 0.001612531300038395
  train_the_pile_pile_cc: 0.8701291419934237
  train_the_pile_ubuntu_irc: 0.06417728505869834
  train_the_pile_europarl: 2.9166170357771267e-06
  train_the_pile_hackernews: 0.011925517591888925
  train_the_pile_pubmed_abstracts: 0.02424425081714838
  train_the_pile_uspto_backgrounds: 0.0024587749419225434
valid:
  valid_the_pile_pile_cc: 1.0
model_name: tinyllama_1_1b
```

最后，使用预测出的最优混合配比来训练你的模型。把 `optimal_mixture.yaml` 放到 `mixture_config/config_1b` 目录下，然后运行下面的脚本：

```bash
cd model_training
./pretrain_tinyllama_1b.sh optimal_mixture
```

即可得到用最优混合配比训练出的最终模型！

### 成功小贴士

- **数据多样性（Data Diversity）**：确保你的初始数据集覆盖足够广泛的领域。
- **代理模型规模（Proxy Model Size）**：虽然我们使用 1M 参数的模型，但你可能需要根据自身的计算资源和数据集规模进行调整。
- **评估（Evaluation）**：选择正确的目标对通用下游性能提升至关重要。你可能希望使用类似 Pile-CC 这样"高质量且多样"的验证数据集上的损失。我们还推荐使用 AI2 出品的出色的 [paloma 评估套件](https://huggingface.co/datasets/allenai/paloma) 进行评估。

### 定制选项

RegMix 非常灵活，可以按你的具体需求进行调整：
- 调整代理模型的数量与规模。
- 修改回归模型的架构或特征。
- 在优化目标中加入领域特定的指标。

请记住，RegMix 成功的关键在于捕捉"数据混合与模型表现"之间的关系。你的代理训练样本越有信息量，最终预测出的混合配比就越好！

## 📦 数据与模型发布

我们已将数据和训练好的模型发布在 HuggingFace 上！

### 模型

以下是完整模型列表，你可以用以下代码加载每个模型：

```python
from transformers import AutoModel, AutoTokenizer

model_name, revision = "sail/data-mixture-random-1b", "model-index-1"
model = AutoModel.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
```

每个模型的详细名称与 revision 如下：

| 模型名称 | Revisions | 描述 | 链接 |
|------------|-----------|-------------|------|
| sail/data-mixture-random-1b | `model-index-1` 至 `model-index-64` | 64 个使用随机数据混合训练的模型，用于研究"数据混合与下游表现"之间的相关性 | [🤗 Hugging Face](https://huggingface.co/sail/data-mixture-random-1b) |
| sail/data-mixture-human-1b | `seed-1` 至 `seed-5` | 5 个使用人工挑选数据混合（基线）的模型，使用不同随机种子 | [🤗 Hugging Face](https://huggingface.co/sail/data-mixture-human-1b) |
| sail/data-mixture-doremi-1b | `seed-1` 至 `seed-5` | 5 个使用 DoReMi 最佳数据混合（基线）的模型，使用不同随机种子 | [🤗 Hugging Face](https://huggingface.co/sail/data-mixture-doremi-1b) |
| sail/data-mixture-pile-cc-1b | `seed-1` 至 `seed-5` | 5 个仅使用 Pile-CC 数据混合训练的模型，使用不同随机种子 | [🤗 HuggingFace](https://huggingface.co/sail/data-mixture-pile-cc-1b) |
| sail/data-mixture-regmix-1b | `seed-1` 至 `seed-5` | 5 个使用 RegMix 数据混合训练的模型，使用不同随机种子 | [🤗 HuggingFace](https://huggingface.co/sail/data-mixture-regmix-1b) |

### 数据

我们还在 HuggingFace 上提供了完整数据集和样本数据，供你参考。你可以手动下载，也可以使用下面的代码下载：

```python
from huggingface_hub import snapshot_download

# 你可以选择下载 regmix-data，或者 regmix-data-sample
snapshot_download(repo_id="sail/regmix-data-sample",
                  repo_type='dataset',
                  local_dir="sail/regmix-data-sample",
                  local_dir_use_symlinks=False)
```

这两个数据集的一些细节如下：

| 数据集名称 | 描述 | 大小 | 链接 |
|--------------|-------------|------|------|
| sail/regmix-data | RegMix 完整数据集，从 [pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) 重新切分而来 | 250B tokens（约 1TB 磁盘空间） | [🤗 Hugging Face](https://huggingface.co/datasets/sail/regmix-data) |
| sail/regmix-data-sample | 从 regmix-data 中抽取的样本数据集，每个领域各保留一个文件 | 5B tokens（约 20GB 磁盘空间） | [🤗 Hugging Face](https://huggingface.co/datasets/sail/regmix-data-sample) |

## 🔍 评估（开发中）

敬请期待！我们目前正在 `evaluation` 目录中提供一套完整的评估方案。

## 📚 引用（Citation）

如果 RegMix 对你的研究有帮助，请引用我们的论文：

```bibtex
@article{liu2024regmix,
  title={RegMix: Data Mixture as Regression for Language Model Pre-training},
  author={Liu, Qian and Zheng, Xiaosen and Muennighoff, Niklas and Zeng, Guangtao and Dou, Longxu and Pang, Tianyu and Jiang, Jing and Lin, Min},
  journal={arXiv preprint arXiv:2407.01492},
  year={2024}
}
```

## 🤝 联系我们

对 RegMix 感兴趣，或有疑问？我们很乐意听到你的声音！

联系方式：
- liuqian@sea.com
- xszheng.2020@phdcs.smu.edu.sg

加入我们，一起迈向可扩展、高效的数据混合之路——与 RegMix 一起！