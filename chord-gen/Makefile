PYTHON    := python3
VENV      := venv
PY        := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip

CLEANED   := dataset/cleaned
DATA      := nanoGPT/data/chords
GPT_OUT   := nanoGPT/out-CHORDv2
RNN_OUT   := nanoGPT/out-CHORDrnn

# Pass extra training flags on the command line, e.g.:
#   make train TRAIN_ARGS="--device=cpu --compile=False"
TRAIN_ARGS ?=

.PHONY: all setup data train train-rnn sample clean help

all: help

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@printf "\nchord-gen — chord progression model pipeline\n\n"
	@printf "  make setup      Create venv and install Python dependencies\n"
	@printf "  make data       Download and prepare the training dataset (~30 min first run)\n"
	@printf "  make train      Train the GPT chord model (saves to $(GPT_OUT))\n"
	@printf "  make train-rnn  Train the RNN/LSTM baseline (saves to $(RNN_OUT))\n"
	@printf "  make sample     Generate sample chord progressions from the trained model\n"
	@printf "  make clean      Remove generated data and checkpoints (keeps venv)\n\n"
	@printf "CPU/MPS override:\n"
	@printf "  make train TRAIN_ARGS=\"--device=cpu --compile=False\"\n\n"

# ── 1. Environment ────────────────────────────────────────────────────────────
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $@

# ── 2. Dataset ────────────────────────────────────────────────────────────────
# Order-only prerequisite (| ...) ensures the venv exists before any data step,
# but doesn't re-trigger the data step when the venv is refreshed.
data: $(DATA)/train.bin

$(CLEANED)/chordonomicon_clean.parquet: dataset/clean_chordonomicon.py | $(VENV)/bin/activate
	$(PY) dataset/clean_chordonomicon.py --outdir $(CLEANED)

$(DATA)/train.bin: $(CLEANED)/chordonomicon_clean.parquet dataset/prepare.py | $(VENV)/bin/activate
	$(PY) dataset/prepare.py \
		--data-dir $(CLEANED) \
		--outdir $(DATA)

# ── 3. Training ───────────────────────────────────────────────────────────────
# cd into nanoGPT/ before running so configurator.py and relative checkpoint
# paths resolve correctly. $(CURDIR) is the absolute path of chord-gen/.
train: $(DATA)/train.bin | $(VENV)/bin/activate
	cd nanoGPT && "$(CURDIR)/$(PY)" train.py config/train_chords.py \
		--out_dir=out-CHORDv2 \
		--doc_aware_batching=True \
		$(TRAIN_ARGS)

train-rnn: $(DATA)/train.bin | $(VENV)/bin/activate
	cd nanoGPT && "$(CURDIR)/$(PY)" train_rnn.py \
		--dataset=chords \
		--out_dir=out-CHORDrnn \
		--max_iters=20000 \
		--hidden=850 \
		$(TRAIN_ARGS)

# ── 4. Sampling ───────────────────────────────────────────────────────────────
sample: | $(VENV)/bin/activate
	@test -f $(GPT_OUT)/ckpt.pt || \
		(printf "\nNo checkpoint at $(GPT_OUT)/ckpt.pt — run 'make train' first.\n\n" && exit 1)
	cd nanoGPT && "$(CURDIR)/$(PY)" sample.py \
		--out_dir=out-CHORDv2 \
		--mode=generate \
		--genre=pop \
		--decade=2010 \
		--num_samples=3

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	rm -rf $(CLEANED) $(DATA) $(GPT_OUT) $(RNN_OUT)
