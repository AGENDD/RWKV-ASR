## 使用预训练的 RWKV 语言模型进行语音识别

- [中文说明](README_CN.md)
- [English](README.md)

本仓库是一个探索性实验，旨在使预训练的 RWKV 语言模型能够接受语音输入。通常，在文本数据上训练的 LLM 不直接适用于语音识别任务，有很多解决方案（例如适配器 + 预训练音频编码器或神经音频编解码器）可以弥合文本和语音之间的差距。我们遵循了 [SLAM_ASR](https://arxiv.org/abs/2402.08846) 的思路，使用 RWKV 语言模型作为 LLM，而不是直接编写提示模板，我们直接微调了 RWKV 模型的初始状态。在 Librispeech 960h Clean 测试集上，我们使用 3B RWKV 模型实现了 4.6% 的 WER（Other 测试集为 6.9%）。

本仓库的代码基于 [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) 开发。当前的语音编码器和适配器实现基于 [SLAM_ASR](https://arxiv.org/abs/2402.08846#)。

### 路线图

我们希望探索计算效率高、性能优越的方式将基于文本的 RWKV 扩展到多模态模型。在音频和语音领域，我们正在尝试以下任务：

- [x] 单语言 ASR
- [ ] 多语言 ASR（即将推出）
- [ ] 语音翻译
- [ ] 语音输入问答（如 GPT-4o）

### 环境

以下命令将创建一个新的 conda 环境并安装所需的包：

```bash
conda create -n rwkv python=3.10
conda activate rwkv
pip install -r requirements.txt
```

### 训练

1. 从以下链接之一下载 RWKV-6-World 模型文件。我们在实验中使用了 3B 模型，即 RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth。

- [Hugging Face](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main)
- [Hf Mirror (CN)](https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main)
- [Modelscope](https://modelscope.cn/models/Blink_DL/rwkv-6-world/files)

2. 打开 ```demo/demo-state-tuning.sh```。将 ```OP=train``` 设置为训练，并将 ```load_model=path/to/your/model/``` 设置为您的模型路径。根据以下表修改 ```n_layer``` 和 ```n_embd```：

|   模型         | n_layer | n_embd  |
| --------- | ---- | ---- |
| 1.6B | 24 | 2048 |
| 3B | 32 | 2560 |
| 7B | 32 | 4096 |
| 14B | 61 | 4096 |

其他训练参数：
|   参数       | 描述  |
| --------- | ---- |
| micro_bsz | 每个设备的批量大小 |
| epoch_steps | 每个 epoch 的步骤数。请根据（数据集大小 / 实际批量大小）进行修改 |
| device | 用于训练的 GPU 数量 |

默认设置将在 4 个设备上训练 3B rwkv 模型，每个设备的批量大小为 4（实际批量大小 = 16）。

3. 该脚本将覆盖 ```output/``` 中的 .pth 文件。确保在训练前将所需的 .pth 模型文件保存到其他目录下！
4. 运行 ```sh demo/demo-state-tuning.sh``` 以开始训练过程。

训练过程如下：

- 它首先加载RWKV模型和从huggingface下载的语音编码模型。将随机初始化适配器和 RWKV 模型的初始状态。
- 模型的（符号）简化公式如下：

```
RWKV( [InitialState], [Adapter](SpeechEncoder(audio))) -> "The weather is good.
```

用`[  ]`包围的部分会被训练，其他参数是锁定的。

还有一些代码可以启用整个模型的其他 PEFT 训练。目前，我们还没有完全适配于语音模态训练，我们仍在积极开发中。

### 评估

参考训练的步骤，但设定`demo/demo-state-tuning.sh`里的`OP=eval`。保存在`output/`中的模型将被用于评估，脚本会计算Librispeech 960h Clean和Other测试集的WER。

### 预训练权重

下载链接即将更新。