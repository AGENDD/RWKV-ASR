## Speech missions with frozen RWKV language models

- [中文说明](README_CN.md)
- [English](README.md)

This repo is an exploratory experiment to enable frozen pretrained RWKV language models to accept speech modality input. Generally, LLMs trained on text data are not directly applicable to speech recognition tasks, and there are many solutions (such as adapters + pretrained audio encoders, or neural audio codecs) to bridge the gap between text and speech. We followed the idea of [SLAM_ASR](https://arxiv.org/abs/2402.08846) and used the RWKV language model as the LLM, and instead of directly writing a prompt template we directly finetuned the initial state of the RWKV model. We were able to achieve 4.6% WER on Librispeech 960h Clean test set (6.9% on Other test) with a 3B RWKV model.

This code inside is developed on [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT). And the current implementation of speech encoder and adapter is based on [SLAM_ASR](https://arxiv.org/abs/2402.08846#).

### Roadmap

We want to explore compute-efficient and high-performance ways to extend text-based RWKV into  multimodal ones. In the audio and speech modality, these are the tasks we are attempting:

- [x] ASR in single language
- [x] ASR in many languages
- [x] Speech Translation
- [x] Voice input question answering (like GPT-4o)
- [ ] Other audio missions
- [ ] Multiple rounds answering

### Environment

The following command will create a new conda environment and install the required packages:

```bash
conda create -n rwkv python=3.10
conda activate rwkv
pip install -r requirements.txt
```

### Training

1. Download RWKV-6-World model files from one of the following links. We used the 3B model in our experiments, i.e. RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth.

- [Hugging Face](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main) 
- [Hf Mirror (CN)](https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main) 
- [Modelscope](https://modelscope.cn/models/Blink_DL/rwkv-6-world/files)

2. Open ```demo/demo-state-tuning.sh```. Set ```OP=train``` for training and ```load_model=path/to/your/model/```. Modify ```n_layer``` and ```n_embd``` according to the table below:

|   Model         | n_layer | n_embd  |
| --------- | ---- | ---- | 
| 1.6B | 24 | 2048 | 
| 3B | 32 | 2560 | 
| 7B | 32 | 4096 | 
| 14B | 61 | 4096 |

Other parameters for training:
|   parameter       | description  |
| --------- | ---- |
| micro_bsz | batch size for each device | 
| epoch_steps | num of steps in 1 epoch. please modified as (dataset size / real batch size) | 
| device | num of GPU for training |  

The default setting will train a 3B rwkv model on librispeech 960h dataset, with 4 devices and a batch size of 4 per device (real batch size = 16). 

3. The script will overwrite the .pth file in ```output/```. Make sure to save the needed .pth model files under this path to other dir before the training.
4. run ```sh demo/demo-state-tuning.sh``` to start the training process.

The training process looks like this:

- It first loads the provided RWKV model, and a speech encoder model from huggingface. An adapter and an initial state for RWKV model will be initialized randomly.
- The (symbolically) simplified formula for this model is:

```
RWKV( [InitialState], [Adapter](SpeechEncoder(audio))) -> "The weather is good. <s>"
```

Modules and variables in `[   ]` will be trained, the rest is all frozen. 

There are also some codes to enable other PEFT training of the whole model. Note that not all methods are fully adapted to speech modality training as of now, and we are still actively working on this.

### Evaluation

Follow the instruction in Training, but modify ```OP=eval``` in ```demo/demo-state-tuning.sh```. The trained model in ```output/``` will be used to calculate the WER of the model in ```output/``` on the clean test set and the other test set of Librispeech.

### Audio File Prediction

Open ```demo/demo-predict.sh``` and modify ```file_path=path/to/your/audio/file```. Run ```sh demo/demo-predict.sh``` to load trained weights in ```output/``` and predict the content of the input audio file.

### Pretrained weights

Download the pretrained weights from the following link:

ASR:https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ASR

SpeechTranslate:https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/ST

SpeechQA:https://huggingface.co/JerryAGENDD/RWKV-ASR/tree/main/SpeechQA

The pretrained weights contain the necessary parameters for the adapter and the RWKV initial state. These weights are trained using WavLM Large as the speech encoder and RWKV-3B as the language model (script default configuration). Place the weights in the ```output/``` directory for the script to load them.

### Speech Chat with RWKV

A script for real-time speech conversation with rwkv. To be continue