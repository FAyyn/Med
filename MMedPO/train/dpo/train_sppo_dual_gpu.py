import os
import sys
import math
import torch
import argparse
from transformers import AutoTokenizer

# Reuse trainer with SPPO variants already implemented
from .dpo_trainer_weighted import DPOTrainer
from ..llava.llava_trainer_weighted import LLaVATrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Dual-GPU Dual-Model SPPO training")
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--ref_model_name_or_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--loss_variant', type=str, default='sppo', choices=['sppo','sppo_adv_squared','tie_sppo','tie_sppo_dynamic'])
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--sppo_eta', type=float, default=1.0)
    parser.add_argument('--sppo_lambda', type=float, default=0.75)
    parser.add_argument('--sppo_alpha', type=float, default=0.5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, required=False, help='Path to preference dataset')
    parser.add_argument('--policy_gpu', type=int, default=0)
    parser.add_argument('--reference_gpu', type=int, default=1)
    parser.add_argument('--lora_enable', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    # Bind processes to GPUs without DDP: we explicitly place two models on two devices
    policy_device = torch.device(f'cuda:{args.policy_gpu}')
    reference_device = torch.device(f'cuda:{args.reference_gpu}')
    torch.cuda.set_device(policy_device)

    # Tokenizer shared
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    # Load policy and reference models on separate GPUs
    from ..llava.model import LlavaLlamaForCausalLM, LlavaMistralForCausalLM, LlavaMPTForCausalLM
    def load_llava(path, device):
        try:
            model = LlavaLlamaForCausalLM.from_pretrained(path, device_map={"": device})
        except Exception:
            try:
                model = LlavaMistralForCausalLM.from_pretrained(path, device_map={"": device})
            except Exception:
                model = LlavaMPTForCausalLM.from_pretrained(path, device_map={"": device})
        return model

    policy_model = load_llava(args.model_name_or_path, f"cuda:{args.policy_gpu}")
    ref_model = load_llava(args.ref_model_name_or_path, f"cuda:{args.reference_gpu}")

    # Initialize vision modules to ensure CLIP vision tower is loaded
    try:
        if hasattr(policy_model, 'get_model'):
            # Minimal args shim with required fields
            class _ArgsShim:
                def __init__(self, vision_tower='openai/clip-vit-large-patch14-336'):
                    self.vision_tower = vision_tower
                    self.mm_vision_select_layer = -2
                    self.mm_vision_select_feature = 'patch'
                    self.mm_patch_merge_type = 'flat'
                    self.mm_projector_type = 'linear'
            policy_model.get_model().initialize_vision_modules(model_args=_ArgsShim(), fsdp=None)
        if hasattr(ref_model, 'get_model'):
            class _ArgsShim:
                def __init__(self, vision_tower='openai/clip-vit-large-patch14-336'):
                    self.vision_tower = vision_tower
                    self.mm_vision_select_layer = -2
                    self.mm_vision_select_feature = 'patch'
                    self.mm_patch_merge_type = 'flat'
                    self.mm_projector_type = 'linear'
            ref_model.get_model().initialize_vision_modules(model_args=_ArgsShim(), fsdp=None)
    except Exception as e:
        print(f"Warning: failed to initialize vision modules: {e}")

    # Move vision modules explicitly
    def place_vision(model, device_id: int):
        if hasattr(model, 'get_vision_tower'):
            vt = model.get_vision_tower()
            if vt is not None:
                vt.to(f'cuda:{device_id}')
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'mm_projector') and model.get_model().mm_projector is not None:
            model.get_model().mm_projector.to(f'cuda:{device_id}')

    place_vision(policy_model, args.policy_gpu)
    place_vision(ref_model, args.reference_gpu)

    # Build trainer using DualGPUDPOTrainer for proper dual-GPU SPPO training
    from tool.dual_gpu_dpo_trainer import DualGPUDPOTrainer
    
    training_args = argparse.Namespace(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
    )

    # Use DualGPUDPOTrainer which properly handles dual-GPU setup with multimodal sequences
    trainer = DualGPUDPOTrainer(
        model=policy_model,
        tokenizer=tokenizer,
        policy_gpu=args.policy_gpu,
        reference_gpu=args.reference_gpu,
        beta=args.beta,
        loss_type='sigmoid',
        loss_variant=args.loss_variant,
        sppo_eta=args.sppo_eta,
        sppo_lambda=args.sppo_lambda,
        sppo_alpha=args.sppo_alpha,
        reference_model=ref_model,
        args=training_args,
    )

    # Dataset hookup: Expect user-side dataset loader; we skip actual dataset plumbing here.
    # Users can call trainer.train() after setting trainer.train_dataset appropriately.
    print("Dual-GPU SPPO trainer initialized.")
    print(f"Policy model on cuda:{args.policy_gpu}, Reference model on cuda:{args.reference_gpu}")
    print(f"Loss variant: {args.loss_variant}, beta={args.beta}, eta={args.sppo_eta}")

    # If dataset path provided and trainer supports set_dataset, wire it; otherwise just initialize.
    if hasattr(trainer, 'train'):
        # Warn if no dataset provided
        if not args.dataset_path:
            print("No dataset_path provided. Please set trainer.train_dataset before calling train().")
        else:
            print("Dataset integration is project-specific; please adapt loader.")

    # Return trainer to allow programmatic use
    return trainer


if __name__ == '__main__':
    main()