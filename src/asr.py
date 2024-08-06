"""
The main body of the ASR model,

User: <Speech> <Prompt>
Model: <Transcription>
"""

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import List

try:
    from .speech_encoder import SpeechEncoder
except ImportError:
    from speech_encoder import SpeechEncoder


from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import RWKV
# from .lora import LinearWithLoRA
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.strategies import DeepSpeedStrategy
import os, math, gc, importlib
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import time

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

class SLAM_ASR(pl.LightningModule):
    def __init__(
        self,
        args,
        speech_encoder_model_id,#facebook/hubert-base-ls960
        language_model,
        downsample_K=5,
        hidden_dim=2048,
        train_mode="adapter",
        device="cuda",
        token = "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",
    ):
        assert train_mode in ["adapter", "full"]
        super().__init__()
        self.args = args
        self._device = device

        self.language_tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-6-world-1b6",trust_remote_code=True)
        ########################################换成RWKV-PEFT的模型结构

        self.language_model = language_model
        #########################################
        
        
        language_project_dim = args.n_embd
        #3B language_project_dim = 2560 
        #7B language_project_dim = 4096
        
        
        self.speech_encoder = SpeechEncoder(
            speech_encoder_model_id,
            language_project_dim,
            downsample_K=downsample_K,
            hidden_dim=hidden_dim,
            train_mode=train_mode,
            device=device,
        ).to(self._device)

        self.set_gradient(train_mode,'state')

    def gradient_checkpointing_enable(self, **kwargs):
        self.language_model.gradient_checkpointing_enable(**kwargs)

    def set_gradient(self, train_mode,tuning):
        assert train_mode in ["adapter", "full"]

        # call set_gradient for speech encoder
        self.speech_encoder.set_gradient(train_mode)
        
        print("Parameters that require grad:")

        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"    {name}: {param.shape}")

    
    def remove_padding(self, x, mask):
        #根据mask去除speech_output的padding部分
        x_no_padding = []
        # 对于每一个样本和对应的掩码
        for x_i, mask_i in zip(x, mask):
            # 使用掩码来选择非填充部分
            x_i_no_padding = x_i[mask_i.bool()]
            # 将结果添加到列表中
            x_no_padding.append(x_i_no_padding)
        
        return x_no_padding
    
    def concatenate_audio_transcription(self, audio, transcription):
        #将两个二维/三维向量在第二维度拼起来
        result = []
        for sublist1, sublist2 in zip(audio, transcription):
            sub_result = torch.cat((sublist1 ,sublist2), dim=0)
            result.append(sub_result)

        return result
    
    
    def _prepare_input_embeds(
        self, audios: List[float], transcriptions: List[str] = None
    ):
        """
        First, run audios through speech_encoder to get the embeddings and mask
        """


        speech_output, mask = self.speech_encoder(audios)
        mask = mask.to(self._device)
        if transcriptions is not None:
            
            ###########处理prompt_embed ###############################################################################
            
            #去除speech padding
            audio_no_padding = self.remove_padding(speech_output,mask)  
            
            #在speech结尾添加end of audio：#
            end_of_audio = self.language_tokenizer(
                "#",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                end_of_audio = self.language_model.embed(end_of_audio.input_ids)
            audio_no_padding_eoa = []
            for t in audio_no_padding:
                t = torch.cat((t, end_of_audio.squeeze(0)))
                audio_no_padding_eoa.append(t)
            
            #audio mask 左边添加1
            ones = torch.ones(mask.size(0), 1).to(self._device)
            mask =torch.cat((ones, mask), dim=1)
            
            #处理transcription，得到embeded label
            _labels = self.language_tokenizer(
                transcriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            with torch.no_grad():
                # labels_embeds = self.language_model.rwkv.get_input_embeddings()(_labels.input_ids)
                labels_embeds = self.language_model.embed(_labels.input_ids)
            att3 = _labels.attention_mask
            
            #拼接speech和label
            audio_label = self.concatenate_audio_transcription(audio_no_padding_eoa , labels_embeds)
            # print(f"concatenated inputs:\t{len(audio_label)}-{[len(x) for x in audio_label]}")
        
            #对拼接后的内容进行padding
            max_seq = max([len(x) for x in audio_label])
            for i, x in enumerate(audio_label):
                times = max_seq - len(x)
                for _ in range(times):
                    x = torch.cat((x,x[len(x)-1].unsqueeze(0)))
                audio_label[i] = x
            # print(f"padded inputs:\t{len(audio_label)}-{[len(x) for x in audio_label]}")
            
            #转换成tensor
            audio_label = torch.stack(audio_label)
            # print(f"padded inputs tensor:\t{audio_label.shape}")
            prompt_embed = audio_label
            # print()
            
            #####处理prompt_mask ##################################################
            
            # 剔除audio mask 右边的0
            mask_no_zero = []
            for mask_i in mask:
                mask_i_no_zero = mask_i[mask_i != 0]
                mask_no_zero.append(mask_i_no_zero)
            
            # 将audio mask和transcription mask 拼接
            mask_concatenate = self.concatenate_audio_transcription(mask_no_zero, att3)
            
            #向mask 填充0
            max_mask = max([len(x) for x in mask_concatenate])
            for i, x in enumerate(mask_concatenate):
                times = max_mask - len(x)
                for _ in range(times):
                    x = torch.cat((x,torch.tensor([0]).to(self.device)))
                mask_concatenate[i] = x

            #转换成tensor
            mask_concatenate = torch.stack(mask_concatenate)
            prompt_mask = mask_concatenate
            
            # #########处理loss mask #####################################################
            # import torch.nn.functional as F
            # loss_mask = []
            
            # for t in mask_no_zero:
            #     pad_len = max_mask - len(t)
            #     pad = F.pad(t, (0, pad_len), "constant", 0)
            #     loss_mask.append(pad)
            
            # loss_mask = torch.stack(loss_mask)
            # loss_mask = prompt_mask - loss_mask
            
            # print(f"loss mask:\t{loss_mask.shape}")
            
            #########处理true_labels ###################################################
            # print()
            
            # 为transcription 结尾添加 end of sentence：<s>
            transcriptions_eos = []
            for starr in transcriptions:
                starr = starr + "<s>"
                transcriptions_eos.append(starr)
            _labels = self.language_tokenizer(
                transcriptions_eos,
                return_tensors="pt",
                padding=True,
                truncation=True,
                add_special_tokens=False,
            ).to(self.device)
            true_labels = _labels.input_ids

            #在ture label左侧填充audio 长度的-100， 同时在右侧填充-100使batch对齐
            padded_labels = []
            for i,t in enumerate(true_labels):
                back_padding = max_mask - t.shape[0] - audio_no_padding[i].shape[0]
                t = torch.cat(
                    [
                        torch.full(
                            (audio_no_padding[i].shape[0], ),
                            -100,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        t,
                        torch.full(
                            (back_padding, ),
                            -100,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ]
                )
                padded_labels.append(t)
            
            padded_labels = torch.stack(padded_labels)
            true_labels = padded_labels
        else:           
            end_of_audio = self.language_tokenizer(
                "#",
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                end_of_audio = self.language_model.embed(end_of_audio.input_ids)
            
            # print(f"speech output:{speech_output.shape}")
            # print(f"end_of_audio:{end_of_audio.shape}")
            # exit(0)
            speech_output = torch.cat((speech_output, end_of_audio), dim= 1)
            
            prompt_embed = speech_output
            prompt_mask = mask
            true_labels = None
        return prompt_embed, prompt_mask, true_labels

    def forward(self, audios: List[float], transcriptions: List[str] = None):
        
        prompt_embed, prompt_mask, true_labels = self._prepare_input_embeds(
            audios, transcriptions
        )

        outputs = self.language_model(inputs_embeds=prompt_embed)

        
        return outputs, true_labels, prompt_mask

    def generate(self, audios: List[float], stopping_criteria=None):
        """
        Generate the transcription
        """
        prompt_embed, prompt_mask, _ = self._prepare_input_embeds(audios)
        
        # outputs = self.language_model(
        #     inputs_embeds=prompt_embed,
        #     attention_mask=prompt_mask.bool()
        # )
        self.language_model.to(self._device, dtype=torch.bfloat16)
        outputs = self.language_model.generate(tokenizer= self.language_tokenizer,inputs_embeds=prompt_embed)
        
        return outputs

    def training_step(self, batch, batch_idx):
            args = self.args
            if args.loss_mask:
                idx, targets, mask = batch
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()
                logits = self(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                loss = torch.sum(loss * mask) / sum_mask
            # elif args.my_qa_mask != 1:
            #     idx, targets = batch
            #     logits = self(idx)
            #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                # if '0' in os.environ["RWKV_MY_TESTING"]:
                #     print('logits', logits)
                #     torch.set_printoptions(threshold=10000)
                #     print('idx', idx)
                #     exit(0)
            else:
                
                ##改动
                # idx, transcription = batch
                idx = [item[0] for item in batch]
                transcription = [item[1] for item in batch]
                
                logits, targets, mask = self(idx, transcription)
                mask = mask.view(-1)
                sum_mask = torch.sum(mask).item()
                ######
                
                # idx, targets, mask = batch
                # mask = mask.view(-1)
                # sum_mask = torch.sum(mask).item()
                # # if sum_mask == 0:
                # #     return torch.tensor([0.0], requires_grad=True)

                # logits = self(idx)
                if sum_mask == mask.shape[0]:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                    # print('rank', self.global_rank, 'loss', loss.item())
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
                    # loss_raw = loss
                    loss = torch.sum(loss * mask) / sum_mask

                    # torch.set_printoptions(threshold=10000)
                    # if True: #self.global_rank == 1:
                    #     tmp = ''
                    #     sss = 0
                    #     ccc = 0
                    #     for i in range(mask.shape[0]):
                    #         if mask[i] > 0:
                    #             tmp += str(idx.view(-1)[i].item()) + ','
                    #             sss += loss_raw.view(-1)[i].float().item()
                    #             ccc += 1
                    #     print('rank', self.global_rank, 'loss', loss.item(), 'lavg', sss / ccc)#, 'tmp', tmp, 'input', idx)

            return L2Wrap.apply(loss, logits)
    
    def configure_optimizers(self):
        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
                lr_1x.add(n)
            elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    def return_tokenizer(self):
        return self.language_tokenizer
    
    @property
    def config(self):
        return self.language_model.config
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        
        self._device = value


    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
