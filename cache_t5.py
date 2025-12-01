#!/usr/bin/env python
"""Pre-cache T5 text embeddings to avoid loading the 11GB model during inference."""
import os
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a person is talking")
    parser.add_argument("--n_prompt", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="./pretrained_models/Wan2.1_Encoders")
    parser.add_argument("--output", type=str, default="./cached_t5_embeddings.pt")
    args = parser.parse_args()

    print(f"Loading T5 encoder from {args.ckpt_dir}...")

    from wan.modules.t5 import T5EncoderModel

    # Load T5 on CPU to save GPU memory
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(args.ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.ckpt_dir, "google/umt5-xxl"),
        shard_fn=None
    )

    print(f"Encoding prompt: '{args.prompt}'")
    context = text_encoder([args.prompt], torch.device('cpu'))

    print(f"Encoding null prompt: '{args.n_prompt}'")
    context_null = text_encoder([args.n_prompt], torch.device('cpu'))

    # Save the embeddings
    cache_data = {
        'prompt': args.prompt,
        'n_prompt': args.n_prompt,
        'context': context,
        'context_null': context_null,
    }

    torch.save(cache_data, args.output)
    print(f"Saved cached embeddings to {args.output}")
    print(f"Context shape: {[c.shape for c in context]}")
    print(f"Context null shape: {[c.shape for c in context_null]}")

if __name__ == "__main__":
    main()
