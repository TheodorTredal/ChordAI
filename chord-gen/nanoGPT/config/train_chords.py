# =============================================================================
# nanoGPT training config: Chordonomicon chord-progression generator
# =============================================================================
# Conditional chord-progression model. Sequences look like:
#   <bos> <genre:pop_rock> <decade:2010> <verse> C G Amin F <chorus> ... <eos>
# Vocabulary ~851 tokens (genre + decade + section + chord + special tokens),
# read automatically from data/chords/meta.pkl.
#
# Dataset scale (after cleaning):
#   ~269,905 training songs  ->  ~23.16M training tokens
#   ~29,988  validation songs ->  ~2.57M validation tokens
#
# Tokens per iteration = batch_size * block_size * grad_accum = 64*256*1 = 16,384
# One epoch over the training set ~= 23.16M / 16,384 ~= 1,414 iterations.
# So max_iters below is roughly 14 epochs; we rely on eval + checkpoint-on-best
# to find the real stopping point rather than trusting the ceiling.
# =============================================================================

# --- I/O ---------------------------------------------------------------------
# out_dir holds CHECKPOINTS and is deliberately separate from data/chords (data).
out_dir = 'out-chords'

# Evaluate often enough to track the val curve and catch the overfit turn.
# At ~1,414 iters/epoch, evaluating every 250 iters gives ~5-6 checks/epoch.
eval_interval = 250
eval_iters = 200          # batches averaged per eval; 200 is a stable estimate
log_interval = 10         # how often to print the training-loss heartbeat

# We have a real-sized dataset (not tiny Shakespeare), so overfitting is NOT
# immediate. Save only when validation loss improves, so the final checkpoint
# is the best-generalizing one rather than the last (possibly overfit) step.
always_save_checkpoint = False

# --- logging -----------------------------------------------------------------
wandb_log = False         # flip to True (and set entity) if you use W&B
wandb_project = 'chordonomicon'
wandb_run_name = 'gpt-chords-baseline'

# --- data --------------------------------------------------------------------
dataset = 'chords'        # -> nanoGPT reads data/chords/{train,val}.bin + meta.pkl
gradient_accumulation_steps = 1
batch_size = 64           # sequences per micro-step
block_size = 256          # context window; covers p95=152 + prefix with headroom

# --- model (a small, fully-from-scratch GPT) --------------------------------
# ~10.7M params at these settings. Small vocab + this depth/width is a good fit
# for ~23M tokens: enough capacity to learn genre/decade/section structure
# without the overfitting a larger net would invite on a dataset this size.
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2             # mild regularization; see note below on tuning it
bias = False              # slightly cleaner/faster; standard modern choice

# --- optimizer / schedule ----------------------------------------------------
learning_rate = 1e-3      # small models tolerate a higher peak LR
max_iters = 20000         # ~14 epochs; a generous ceiling, not a target
lr_decay_iters = 20000    # decay over the whole run (keep == max_iters)
min_lr = 1e-4             # floor = learning_rate / 10, the usual ratio
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99              # a bit below 0.95-default's sibling 0.999 because the
                          # token count per iter is modest; matches Karpathy's
                          # small-model guidance
grad_clip = 1.0
warmup_iters = 200        # short linear warmup; stabilizes the first few hundred steps

# --- system ------------------------------------------------------------------
# Leave device/dtype to train.py defaults (cuda + bf16 if supported). If your
# cluster GPUs predate bf16 (pre-Ampere), set dtype = 'float16' here.
# compile = True is on by default in train.py and gives a real speedup on
# PyTorch 2.x; if you hit a compile error on the cluster, override with
# --compile=False on the command line rather than editing this file.
