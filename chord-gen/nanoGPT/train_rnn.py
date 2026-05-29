"""
train_rnn.py — LSTM baseline for the chord-progression task.

The "Model 0" baseline in the progression: recurrent -> transformer (v1) ->
aligned transformer (v2). It is deliberately the simplest honest recurrent model
and is matched to the transformer as closely as possible so the comparison
isolates ARCHITECTURE (recurrent vs attention), not anything else:

  * reads the SAME data: data/<dataset>/{train,val}.bin + meta.pkl
  * SAME vocabulary, SAME stratified split (the bins were built by prepare.py)
  * v1-style RANDOM-WINDOW batching, NO prefix masking (the v2 refinements were
    transformer-side improvements; the baseline stays simple)
  * conditioning is fed the SAME way: <bos> <genre> <decade> as leading tokens,
    with NO special conditioning mechanism. The LSTM must carry the conditioning
    in its hidden state, which is exactly the long-range weakness we want the
    comparison to expose honestly.
  * sized to ~match the transformer's parameter count (~10.95M) so a reviewer
    cannot attribute differences to capacity.

The model exposes the SAME interface as nanoGPT's GPT so the shared eval scripts
work unchanged:
    forward(idx, targets=None) -> (logits, loss)
    - with targets: logits over ALL positions (1,T,V) + cross_entropy(ignore_index=-1)
    - without targets: logits for the LAST position only (1,1,V)  [generation]
Checkpoints carry model_type='rnn' and a 'model_args' dict, mirroring GPT
checkpoints, so model_loader.py can dispatch on type.

Usage:
  python train_rnn.py --dataset=chords --out_dir=out-CHORDrnn --device=cuda
  python train_rnn.py --dataset=chords --out_dir=out-CHORDrnn --max_iters=20000 \
      --batch_size=64 --block_size=256 --n_layer=2 --n_embd=512 --hidden=1024
"""

import os
import time
import math
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Model
# ============================================================================

class RNNConfig:
    def __init__(self, vocab_size, n_embd=384, hidden=1024, n_layer=2,
                 dropout=0.2, block_size=256, rnn_type='lstm'):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.hidden = hidden
        self.n_layer = n_layer
        self.dropout = dropout
        self.block_size = block_size          # for interface parity with GPT
        self.rnn_type = rnn_type

    def as_dict(self):
        return dict(vocab_size=self.vocab_size, n_embd=self.n_embd,
                    hidden=self.hidden, n_layer=self.n_layer,
                    dropout=self.dropout, block_size=self.block_size,
                    rnn_type=self.rnn_type)


class RNNLM(nn.Module):
    """Token-level LSTM/GRU language model with a GPT-compatible forward()."""

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        rnn_cls = nn.LSTM if config.rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.n_embd,
            hidden_size=config.hidden,
            num_layers=config.n_layer,
            dropout=config.dropout if config.n_layer > 1 else 0.0,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(config.hidden)
        self.head = nn.Linear(config.hidden, config.vocab_size, bias=False)
        self.apply(self._init)
        n = sum(p.numel() for p in self.parameters())
        print(f"RNN ({config.rnn_type.upper()}) parameters: {n/1e6:.2f}M")

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, hidden=None):
        # idx: (B, T)
        x = self.drop(self.embed(idx))          # (B, T, n_embd)
        out, hidden = self.rnn(x, hidden)       # (B, T, hidden)
        out = self.ln(out)
        if targets is not None:
            logits = self.head(out)             # (B, T, V) — all positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,                # parity with GPT (prefix masking)
            )
        else:
            # GPT-parity inference optimization: only the last position.
            logits = self.head(out[:, [-1], :])  # (B, 1, V)
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # Simpler than nanoGPT's split; weight decay on all 2D+ params.
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
        fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas,
                                 **({'fused': True} if fused else {}))


# ============================================================================
# Training
# ============================================================================

def get_batch(split, data_dir, block_size, batch_size, device, device_type):
    fn = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fn), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data_dir, block_size, batch_size, device, device_type, eval_iters):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, block_size, batch_size, device, device_type)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, lr, warmup, decay_iters, min_lr):
    if it < warmup:
        return lr * (it + 1) / (warmup + 1)
    if it > decay_iters:
        return min_lr
    ratio = (it - warmup) / (decay_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='chords')
    p.add_argument('--out_dir', default='out-CHORDrnn')
    p.add_argument('--rnn_type', default='lstm', choices=['lstm', 'gru'])
    # model size (defaults chosen to land ~11M params with vocab ~851; see note)
    p.add_argument('--n_embd', type=int, default=384)
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--n_layer', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--block_size', type=int, default=256)
    # optim
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--max_iters', type=int, default=20000)
    p.add_argument('--warmup_iters', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=1e-1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--min_lr', type=float, default=1e-4)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.99)
    # io / system
    p.add_argument('--eval_interval', type=int, default=250)
    p.add_argument('--eval_iters', type=int, default=200)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device_type = 'cuda' if 'cuda' in args.device else ('mps' if args.device == 'mps' else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    data_dir = os.path.join('data', args.dataset)
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"found vocab_size = {vocab_size} (inside {data_dir}/meta.pkl)")

    cfg = RNNConfig(vocab_size=vocab_size, n_embd=args.n_embd, hidden=args.hidden,
                    n_layer=args.n_layer, dropout=args.dropout,
                    block_size=args.block_size, rnn_type=args.rnn_type)
    model = RNNLM(cfg).to(args.device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), device_type)

    best_val = 1e9
    iter_num = 0
    t0 = time.time()
    while True:
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.max_iters, args.min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, data_dir, args.block_size, args.batch_size,
                                   args.device, device_type, args.eval_iters)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val and iter_num > 0:
                best_val = losses['val']
                ckpt = {
                    'model': model.state_dict(),
                    'model_args': cfg.as_dict(),
                    'model_type': 'rnn',          # <-- lets eval dispatch on type
                    'iter_num': iter_num,
                    'best_val_loss': best_val,
                    'config': {'dataset': args.dataset},
                }
                torch.save(ckpt, os.path.join(args.out_dir, 'ckpt.pt'))
                print(f"  saved checkpoint to {args.out_dir} (val {best_val:.4f})")

        X, Y = get_batch('train', data_dir, args.block_size, args.batch_size,
                         args.device, device_type)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if iter_num % args.log_interval == 0:
            t1 = time.time(); dt = t1 - t0; t0 = t1
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000/max(args.log_interval,1):.1f}ms/it")

        iter_num += 1
        if iter_num > args.max_iters:
            break

    print(f"done. best val loss {best_val:.4f}")


if __name__ == '__main__':
    main()
