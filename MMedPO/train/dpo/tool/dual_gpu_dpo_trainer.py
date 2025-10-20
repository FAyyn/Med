#!/usr/bin/env python3
"""
双GPU DPO Trainer - 将policy模型和reference模型分别部署在不同GPU上
这样可以避免adapter禁用/启用的问题，提高训练效率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import copy
import warnings
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from peft import PeftModel
import logging
import sys
import os

# 添加父目录到路径以导入DPOTrainer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo_trainer_weighted import DPOTrainer

logger = logging.getLogger(__name__)

class DualGPUDPOTrainer(DPOTrainer):
    """
    双GPU DPO Trainer
    - Policy模型（带LoRA）在GPU 0
    - Reference模型（不带LoRA）在GPU 1
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        policy_gpu: int = 0,
        reference_gpu: int = 1,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        loss_variant: str = "dpo",
        sppo_eta: float = 0.0,
        sppo_lambda: float = 0.0,
        sppo_alpha: float = 1.0,
        reference_free: bool = False,
        reference_model: Optional[PreTrainedModel] = None,
        args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        """
        初始化双GPU DPO Trainer
        
        Args:
            model: 预训练模型（通常是带LoRA的policy模型）
            tokenizer: 分词器
            policy_gpu: policy模型所在的GPU
            reference_gpu: reference模型所在的GPU
            beta: DPO温度参数
            label_smoothing: 标签平滑参数
            loss_type: 损失函数类型
            loss_variant: 损失变体 ('dpo', 'sppo', 'sppo_adv_squared', 'tie_sppo')
            sppo_eta: SPPO eta参数
            sppo_lambda: SPPO lambda参数
            sppo_alpha: SPPO alpha参数
            reference_free: 是否使用reference-free模式
            reference_model: 可选的预加载参考模型
            args: 训练参数
            **kwargs: 其他参数传递给父类
        """
        # 双GPU特定的初始化（在父类初始化之前）
        self.policy_gpu = policy_gpu
        self.reference_gpu = reference_gpu
        self.reference_free = reference_free
        
        # 设置dual-GPU标志，避免父类的accelerator.prepare_model调用
        self.enable_dual_gpu = True
        
        # 设置policy模型（带LoRA）
        self.policy_model = model
        
        # 创建或使用提供的reference模型
        if reference_model is not None:
            self.ref_model = reference_model
            self.ref_model.eval()
            # 冻结参数
            for param in self.ref_model.parameters():
                param.requires_grad = False
            logger.info("使用提供的reference模型")
        else:
            # 创建reference模型（不带LoRA）
            self.ref_model = self._create_reference_model(model)
        
        # 初始化父类DPOTrainer
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            beta=beta,
            loss_type=loss_type,
            loss_variant=loss_variant,
            sppo_eta=sppo_eta,
            enable_dual_gpu=True,  # 传递enable_dual_gpu参数给父类
            policy_gpu=policy_gpu,  # 传递policy_gpu参数
            reference_gpu=reference_gpu,  # 传递reference_gpu参数
            sppo_lambda=sppo_lambda,
            sppo_alpha=sppo_alpha,
            args=args,
            **kwargs
        )
        
        logger.info(f"Policy模型部署在GPU {policy_gpu}")
        logger.info(f"Reference模型部署在GPU {reference_gpu}")
        
    def _create_reference_model(self, policy_model):
        """创建reference模型（不带LoRA适配器）"""
        if isinstance(policy_model, PeftModel):
            # 如果policy模型是PeftModel，获取基础模型
            base_model = policy_model.get_base_model()
            
            # 检查是否是量化模型，如果是则不能使用.to()方法
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'quantization_config'):
                # 对于量化模型，我们需要重新加载到目标设备
                logger.warning("检测到量化模型，将重新加载到reference GPU")
                # 获取模型路径和配置
                model_path = getattr(base_model.config, '_name_or_path', None)
                if model_path is None:
                    # 如果无法获取路径，使用共享权重但不移动设备
                    logger.warning("无法获取模型路径，使用共享权重模式")
                    reference_model = base_model
                else:
                    # 重新加载模型到目标设备
                    from transformers import AutoModelForCausalLM
                    import torch
                    device_map = {"": f'cuda:{self.reference_gpu}'}
                    reference_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
            else:
                # 非量化模型可以直接移动
                reference_model = base_model.to(f'cuda:{self.reference_gpu}')
            
            # 设置为评估模式
            reference_model.eval()
            
            # 冻结参数
            for param in reference_model.parameters():
                param.requires_grad = False
                
            logger.info("创建了不带LoRA的reference模型")
            return reference_model
        else:
            # 如果policy模型不是PeftModel，检查是否是量化模型
            if hasattr(policy_model, 'config') and hasattr(policy_model.config, 'quantization_config'):
                logger.warning("检测到量化的非PEFT模型，使用共享权重模式")
                reference_model = policy_model
            else:
                # 非量化模型可以深拷贝和移动
                reference_model = copy.deepcopy(policy_model)
                reference_model = reference_model.to(f'cuda:{self.reference_gpu}')
            
            reference_model.eval()
            
            for param in reference_model.parameters():
                param.requires_grad = False
                
            logger.info("创建了reference模型")
            return reference_model
    
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """计算批次的对数概率"""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not average_log_prob:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        
        loss_mask = labels != label_pad_token_id
        
        # 确保labels中的值在有效范围内
        vocab_size = logits.shape[-1]
        labels = torch.clamp(labels, 0, vocab_size - 1)
        
        # 对于-100的位置，设置为0（但会被loss_mask过滤掉）
        labels = torch.where(labels == label_pad_token_id, 0, labels)

        # 计算对数概率
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, 
        model: torch.nn.Module, 
        batch: Dict[str, torch.Tensor],
        device: str
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        在指定设备上执行模型前向传播，处理多模态序列的可变长度
        
        Args:
            model: 要使用的模型
            batch: 输入批次
            device: 目标设备
            
        Returns:
            chosen_logps, rejected_logps, chosen_logits, rejected_logits
        """
        # 确保模型在正确的设备上（跳过量化模型）
        if not (hasattr(model, 'config') and hasattr(model.config, 'quantization_config') and 
                model.config.quantization_config is not None):
            model = model.to(device)
        
        # 将批次数据移动到指定设备
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device)
            else:
                batch_on_device[key] = value
        
        # 处理chosen数据
        chosen_outputs = model(
            input_ids=batch_on_device['chosen_input_ids'],
            attention_mask=batch_on_device['chosen_attention_mask'],
            images=batch_on_device.get('images', None)
        )
        chosen_logits = chosen_outputs.logits
        
        # 使用multimodal准备的labels
        if hasattr(model, 'module'):
            underlying_model = model.module
        else:
            underlying_model = model
            
        if hasattr(underlying_model, 'prepare_inputs_labels_for_multimodal'):
            _, _, _, _, _, new_chosen_labels = underlying_model.prepare_inputs_labels_for_multimodal(
                input_ids=batch_on_device['chosen_input_ids'],
                position_ids=None,
                attention_mask=batch_on_device['chosen_attention_mask'],
                past_key_values=None,
                labels=batch_on_device['chosen_labels'],
                images=batch_on_device.get('images', None)
            )
            chosen_logps = self._get_batch_logps(chosen_logits, new_chosen_labels, average_log_prob=True)
        else:
            chosen_logps = self._get_batch_logps(chosen_logits, batch_on_device['chosen_labels'], average_log_prob=True)
        
        # 处理rejected数据
        rejected_outputs = model(
            input_ids=batch_on_device['rejected_input_ids'],
            attention_mask=batch_on_device['rejected_attention_mask'],
            images=batch_on_device.get('rejected_images', batch_on_device.get('images', None))
        )
        rejected_logits = rejected_outputs.logits
        
        if hasattr(underlying_model, 'prepare_inputs_labels_for_multimodal'):
            _, _, _, _, _, new_rejected_labels = underlying_model.prepare_inputs_labels_for_multimodal(
                input_ids=batch_on_device['rejected_input_ids'],
                position_ids=None,
                attention_mask=batch_on_device['rejected_attention_mask'],
                past_key_values=None,
                labels=batch_on_device['rejected_labels'],
                images=batch_on_device.get('rejected_images', batch_on_device.get('images', None))
            )
            rejected_logps = self._get_batch_logps(rejected_logits, new_rejected_labels, average_log_prob=True)
        else:
            rejected_logps = self._get_batch_logps(rejected_logits, batch_on_device['rejected_labels'], average_log_prob=True)
        
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def get_batch_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        train_eval: str = "train",
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        计算批次的DPO损失和指标
        
        Args:
            batch: 输入批次
            train_eval: 训练或评估模式
            
        Returns:
            losses, metrics
        """
        metrics = {}
        
        # 在policy GPU上计算policy模型输出
        with torch.no_grad() if train_eval == "eval" else torch.enable_grad():
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = self.concatenated_forward(
                self.policy_model, 
                batch, 
                f'cuda:{self.policy_gpu}'
            )
        
        # 在reference GPU上计算reference模型输出
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(
                self.ref_model, 
                batch, 
                f'cuda:{self.reference_gpu}'
            )
        
        # 将reference模型的输出移动到policy GPU进行损失计算
        reference_chosen_logps = reference_chosen_logps.to(f'cuda:{self.policy_gpu}')
        reference_rejected_logps = reference_rejected_logps.to(f'cuda:{self.policy_gpu}')
        
        # 计算损失，支持不同的损失变体
        losses, chosen_rewards, rejected_rewards = self._compute_loss_with_variant(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            batch_weight=batch.get('weight', None)
        )
        
        # 计算logratios用于调试信息
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        
        metrics[f"rewards_{train_eval}/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"rewards_{train_eval}/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"rewards_{train_eval}/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().cpu()
        metrics[f"rewards_{train_eval}/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"logps_{train_eval}/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"logps_{train_eval}/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"logits_{train_eval}/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"logits_{train_eval}/chosen"] = policy_chosen_logits.detach().mean().cpu()
        
        # 添加调试信息
        metrics[f"debug_{train_eval}/policy_chosen_logps"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"debug_{train_eval}/policy_rejected_logps"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"debug_{train_eval}/reference_chosen_logps"] = reference_chosen_logps.detach().mean().cpu()
        metrics[f"debug_{train_eval}/reference_rejected_logps"] = reference_rejected_logps.detach().mean().cpu()
        metrics[f"debug_{train_eval}/pi_logratios"] = pi_logratios.detach().mean().cpu()
        metrics[f"debug_{train_eval}/ref_logratios"] = ref_logratios.detach().mean().cpu()
        metrics[f"debug_{train_eval}/beta_logits"] = (self.beta * logits).detach().mean().cpu()
        
        return losses.mean(), metrics
    
    def _compute_loss_with_variant(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        batch_weight: torch.FloatTensor = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        根据损失变体计算损失
        
        Args:
            policy_chosen_logps: 策略模型选择响应的对数概率
            policy_rejected_logps: 策略模型拒绝响应的对数概率
            reference_chosen_logps: 参考模型选择响应的对数概率
            reference_rejected_logps: 参考模型拒绝响应的对数概率
            batch_weight: 批次权重
            
        Returns:
            losses, chosen_rewards, rejected_rewards
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if self.reference_free:
            ref_logratios = 0

        # Base margin between policy and reference
        logits = pi_logratios - ref_logratios

        if self.loss_variant == "sppo":
            # Original SPPO Loss
            logratio_w = (policy_chosen_logps - reference_chosen_logps)
            logratio_l = (policy_rejected_logps - reference_rejected_logps)
            
            # Compute preference probability P(y_w > y_l|x) using sigmoid with temperature β
            preference_logits = self.beta * (logratio_w - logratio_l)
            p_pref = torch.sigmoid(preference_logits)
            
            # Numerical stability: clamp probabilities to avoid saturation
            epsilon = 1e-8
            p_pref = torch.clamp(p_pref, epsilon, 1 - epsilon)
            p_anti = 1 - p_pref
            
            # Compute SPPO loss terms with η scaling
            term_chosen = logratio_w - self.sppo_eta * (p_pref - 0.5)
            term_rejected = logratio_l - self.sppo_eta * (p_anti - 0.5)
            
            losses = term_chosen ** 2 + term_rejected ** 2
            
            if self.loss_use_weight and batch_weight is not None:
                losses = losses * batch_weight

            chosen_rewards = self.beta * logratio_w.detach()
            rejected_rewards = self.beta * logratio_l.detach()

            return losses, chosen_rewards, rejected_rewards
            
        elif self.loss_variant == "sppo_adv_squared":
            # SPPO-ADV Loss (Rewritten version with positive reinforcement)
            logratio_w = (policy_chosen_logps - reference_chosen_logps)
            logratio_l = (policy_rejected_logps - reference_rejected_logps)
            
            # Compute preference probability P(y_w > y_l|x) using sigmoid with temperature β
            preference_logits = self.beta * (logratio_w - logratio_l)
            p_pref = torch.sigmoid(preference_logits)
            
            # Numerical stability: clamp probabilities to avoid saturation
            epsilon = 1e-8
            p_pref = torch.clamp(p_pref, epsilon, 1 - epsilon)
            p_anti = 1 - p_pref
            
            # Compute SPPO-ADV loss terms with λ weighting for positive reinforcement
            term_chosen = logratio_w - self.sppo_eta * (p_pref - 0.5)
            term_rejected = logratio_l - self.sppo_eta * (p_anti - 0.5)
            
            losses = self.sppo_lambda * (term_chosen ** 2) + (1.0 - self.sppo_lambda) * (term_rejected ** 2)

            # Apply optional dataset-level weights
            if self.loss_use_weight and batch_weight is not None:
                losses = losses * batch_weight

            chosen_rewards = self.beta * logratio_w.detach()
            rejected_rewards = self.beta * logratio_l.detach()

            return losses, chosen_rewards, rejected_rewards
            
        elif self.loss_variant == "tie_sppo":
            # TIE-SPPO Loss (Basic version with fixed weighting)
            logratio_w = (policy_chosen_logps - reference_chosen_logps)
            logratio_l = (policy_rejected_logps - reference_rejected_logps)
            
            # Clamp log ratios to prevent extreme values
            logratio_w = torch.clamp(logratio_w, -10.0, 10.0)
            logratio_l = torch.clamp(logratio_l, -10.0, 10.0)
            
            # Compute preference probability P(y_w > y_l|x) using sigmoid with temperature β
            preference_logits = self.beta * (logratio_w - logratio_l)
            p_pref = torch.sigmoid(preference_logits)
            
            # Numerical stability: clamp probabilities to avoid saturation
            epsilon = 1e-8
            p_pref = torch.clamp(p_pref, epsilon, 1 - epsilon)
            p_anti = 1 - p_pref
            
            # Compute TIE-SPPO loss terms with fixed weighting using α parameter
            term_chosen = logratio_w - self.sppo_eta * (p_pref - 0.5)
            term_rejected = logratio_l - self.sppo_eta * (p_anti - 0.5)
            
            # Use α as fixed weight parameter
            losses = self.sppo_alpha * (term_chosen ** 2) + (1.0 - self.sppo_alpha) * (term_rejected ** 2)
            
            chosen_rewards = self.beta * logratio_w.detach()
            rejected_rewards = self.beta * logratio_l.detach()

            # Apply optional dataset weights if enabled and provided
            if self.loss_use_weight and batch_weight is not None:
                losses = losses * batch_weight

            return losses, chosen_rewards, rejected_rewards
            
        else:
            # Default DPO loss
            if self.loss_type == "sigmoid":
                losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            elif self.loss_type == "hinge":
                losses = torch.relu(1 - self.beta * logits)
            elif self.loss_type == "ipo":
                losses = (logits - 1/(2 * self.beta)) ** 2
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
            
            return losses, chosen_rewards, rejected_rewards
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        batch: Dict[str, torch.Tensor],
        train_eval: str = "train",
    ) -> torch.FloatTensor:
        """计算DPO损失"""
        # 如果model被DataParallel包装，提取原始模型
        if hasattr(model, 'module'):
            model = model.module
        
        loss, _ = self.get_batch_metrics(batch, train_eval)
        return loss
    
    def get_eval_metrics(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.FloatTensor]:
        """获取评估指标"""
        _, metrics = self.get_batch_metrics(batch, "eval")
        return metrics
    
    def save_model(self, output_dir: str, _internal_call: bool = False):
        """保存模型"""
        # 只保存policy模型（包含LoRA适配器）
        if hasattr(self.policy_model, 'save_pretrained'):
            self.policy_model.save_pretrained(output_dir)
        else:
            torch.save(self.policy_model.state_dict(), f"{output_dir}/pytorch_model.bin")
        
        # 保存tokenizer
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(output_dir)
            
        logger.info(f"模型已保存到 {output_dir}")
    
    def __del__(self):
        """清理GPU内存"""
        try:
            if hasattr(self, 'policy_model'):
                del self.policy_model
            if hasattr(self, 'reference_model'):
                del self.reference_model
            torch.cuda.empty_cache()
        except:
            pass