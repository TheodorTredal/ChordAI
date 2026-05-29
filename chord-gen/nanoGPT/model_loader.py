"""
model_loader.py — Load either a GPT (nanoGPT) or RNN checkpoint uniformly.

Both model classes expose the same forward(idx, targets=None) -> (logits, loss)
contract, so once loaded they are interchangeable for evaluation/generation.
This loader dispatches on the checkpoint's 'model_type' field:
  * 'rnn'             -> RNNLM   (from train_rnn.py)
  * 'gpt' or missing  -> GPT     (from model.py; existing v1/v2 checkpoints
                                  predate the field, so absence => gpt)

Returns (model, checkpoint) with the model in eval mode on the given device.
"""

import torch


def load_any(out_dir, device):
    import os
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    model_type = ckpt.get('model_type', 'gpt')

    if model_type == 'rnn':
        from train_rnn import RNNConfig, RNNLM
        model = RNNLM(RNNConfig(**ckpt['model_args']))
    else:
        from model import GPTConfig, GPT
        model = GPT(GPTConfig(**ckpt['model_args']))

    state_dict = ckpt['model']
    for k in list(state_dict.keys()):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, ckpt, model_type
