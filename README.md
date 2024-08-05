This repo is developed on [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT). It follows the framwork of SLAM_ASR and opens the state tuning for RWKV models. This realizes high-accuracy and instruction-free ASR based on RWKV language model.


# Environment

# Training

Download RWKV-6-World model files: 

- [Hugging Face](https://huggingface.co/BlinkDL/rwkv-6-world/tree/main) 
- [Hf Mirror (CN)](https://hf-mirror.com/BlinkDL/rwkv-6-world/tree/main) 
- [Modelscope](https://modelscope.cn/models/Blink_DL/rwkv-6-world/files)

Open ```demo/demo-state-tuning.sh```. Modify ```OP=1``` for training and ```load_model=path/to/your/model/```. Modify ```n_layer``` and ```n_embd``` according to the table below:

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

The script will overwrite the .pth file in ```output/```. Make sure to save the needed .pth model files under this path to other dir before the training.

run ```sh demo/demo-state-tuning.sh``` to start the script.




# Predition

Follow the instruction in Traning, but modify ```OP=2``` in ```demo/demo-state-tuning.sh```. The model in ```output/``` will predicts 100 examples in Librispeech 960h.

# WER test

Follow the instruction in Traning, but modify ```OP=3``` in ```demo/demo-state-tuning.sh```. The script will calculate the WER of the model in ```output/``` the clean test set and the other test set of Librispeech.