"""Fine-tune Muse Glimmer on Modal and serve it next to the base model.

    modal run scripts/finetune/modal_app.py::download            # weights -> a Modal volume, once
    modal run scripts/finetune/modal_app.py --smoke              # 2 steps on 16 examples: memory, speed
    modal run scripts/finetune/modal_app.py                      # QLoRA on runs/finetune/train.jsonl
    modal deploy scripts/finetune/modal_app.py                   # vLLM: base + adapter, one endpoint
    modal app stop epicenter-glimmer                             # stop paying for the GPU

Training follows the QLoRA recipe in Meta's Muse Glimmer post-training guide
(4-bit NF4 base, LoRA adapters on the language model's projections, bf16
compute) with transformers and PEFT. The examples are tokenized with Glimmer's
own chat template and carry their label masks (sft_tokens.py), and the loss is
computed only where the model writes, which is also where the logits are
materialized: long tool outputs cost attention, not a vocabulary-sized logit
tensor.

Serving uses Meta's vLLM image for Glimmer with its tool-call and reasoning
parsers. One server holds the base model (`muse-glimmer`) and the trained
adapter (`glimmer-tuned`), so base and tuned are evaluated on the same
weights, server and settings. The endpoint requires GLIMMER_API_KEY, read from
the local .env at deploy time.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

MODEL = "meta-models/Muse-Glimmer-30B"
REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"      # the revision Meta's cookbook pins
RUN = "glimmer-rca-v1"
CHECKPOINT_EVERY = 9
MAX_LEN = 32768          # training; every example fits
SERVE_LEN = 65536        # serving: the model's own reasoning accumulates turn by turn
HERE = Path(__file__).resolve().parent

app = modal.App("epicenter-glimmer")
weights = modal.Volume.from_name("glimmer-weights", create_if_missing=True)
runs = modal.Volume.from_name("glimmer-runs", create_if_missing=True)
ENV = {"HF_HOME": "/weights", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.12.1", extra_options="--index-url https://download.pytorch.org/whl/cu130")
    # the library versions Meta's recipe was verified with (Axolotl 0.19.0's pins)
    .uv_pip_install("transformers==5.16.1", "peft==0.20.0", "bitsandbytes==0.50.2", "accelerate==1.13.0",
                    "huggingface_hub==1.17.0")
    .env(ENV)
    .add_local_file(HERE / "sft_tokens.py", "/root/sft_tokens.py")
)
# Meta's vLLM image for Glimmer (the muse-glimmer tag), pinned to its linux/amd64 digest. Its Python is
# installed by uv as python3 only; Modal looks for `python`.
VLLM_IMAGE = "vllm/vllm-openai@sha256:dbb04f5c0c0a6356779cf3175381a0406584df01b0ad2f88eaadb46f7f2da08c"
vllm_image = modal.Image.from_registry(
    VLLM_IMAGE, setup_dockerfile_commands=['RUN ln -sf "$(command -v python3)" /usr/local/bin/python'],
).entrypoint([]).env({**ENV, "HF_HUB_OFFLINE": "1"})


def _api_key() -> dict:
    if not modal.is_local():
        return {}
    import sys
    sys.path.insert(0, str(HERE.parents[1]))
    from src.llm import api_key
    return {"GLIMMER_API_KEY": api_key("GLIMMER_API_KEY")}


@app.function(image=train_image, volumes={"/weights": weights}, timeout=3600, cpu=4)
def download():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                                       "*.txt", "*.model"])
    weights.commit()
    print("weights at", path)


@app.function(image=train_image, gpu="H100", volumes={"/weights": weights, "/runs": runs}, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def train(train_rows: list[dict], validation_rows: list[dict], hparams: dict):
    import math
    import random
    import time

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig,
                              get_cosine_schedule_with_warmup)

    from sft_tokens import tokenize

    out = Path("/runs") / (RUN + ("-smoke" if hparams.get("max_steps") else ""))
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    data = {}
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        encoded = [tokenize(tokenizer, r, MAX_LEN) for r in rows]
        data[name] = [e for e in encoded if e is not None]
        print(f"{name}: {len(data[name])} examples ({len(rows) - len(data[name])} over {MAX_LEN} tokens dropped), "
              f"{sum(len(e['input_ids']) for e in data[name])} tokens, "
              f"{sum(sum(l != -100 for l in e['labels']) for e in data[name])} trained")

    # Glimmer is registered as an image-text-to-text model (MuseGlimmerForConditionalGeneration);
    # training is text-only and the vision tower stays frozen.
    # Three of every four layers use 2,048-token sliding-window attention. Under SDPA that window is an
    # explicit mask, which sends PyTorch to its math kernel and materializes seq^2 scores (70 GiB at 24k
    # tokens). FlexAttention runs the window as a block-sparse mask instead.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=REVISION, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="flex_attention",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                               bnb_4bit_quant_storage=torch.bfloat16))
    # Not peft's prepare_model_for_kbit_training: it upcasts the norms to fp32, which makes every
    # activation fp32 too. Activations stay bf16, as in Meta's recipe.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=hparams["rank"], lora_alpha=2 * hparams["rank"], lora_dropout=0.05, task_type="CAUSAL_LM",
        # Meta's recipe: the language model's projections only. Glimmer's attention also has a
        # gate_proj of its own, which this leaves alone, as does the vision tower.
        target_modules=r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
    model.print_trainable_parameters()
    lora_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("adapted, e.g.:", lora_names[:2], "...", lora_names[-1])

    def loss_of(example) -> tuple[torch.Tensor, int]:
        ids = torch.tensor([example["input_ids"]], device="cuda")
        labels = torch.tensor(example["labels"][1:], device="cuda")
        keep = (labels != -100).nonzero().squeeze(1)          # positions whose next token is trained
        logits = model(input_ids=ids, logits_to_keep=keep).logits[0]
        return F.cross_entropy(logits.float(), labels[keep], reduction="sum"), len(keep)

    def evaluate() -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for ex in data["validation"]:
                loss, n = loss_of(ex)
                total, count = total + loss.item(), count + n
        model.train()
        return total / max(count, 1)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hparams["lr"], weight_decay=0.0)
    accum, epochs = hparams["accumulation"], hparams["epochs"]
    per_epoch = math.ceil(len(data["train"]) / accum)
    steps = epochs * per_epoch
    max_steps = hparams.get("max_steps") or steps
    schedule = get_cosine_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    # A checkpoint every CHECKPOINT_EVERY steps (adapter, optimizer, schedule, log) lets a cancelled or
    # preempted run resume where it stopped. Each epoch's order is seeded by the epoch alone, so the
    # resumed run skips exactly the batches already done.
    ckpt = out / "checkpoint.pt"
    trainable = lambda: {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}  # noqa: E731
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["adapter"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        step, log = state["step"], state["log"]
        log.setdefault("resumed_at", []).append(step)
        print(f"resumed from step {step}")
    else:
        step = 0
        log = {"hparams": hparams, "max_len": MAX_LEN, "model": MODEL, "revision": REVISION,
               "examples": {k: len(v) for k, v in data.items()}, "steps": [], "validation": []}
        log["validation"].append({"step": 0, "loss": evaluate()})
        print("validation loss before training:", log["validation"][-1]["loss"])

    def save_checkpoint():
        torch.save({"adapter": trainable(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                    "step": step, "log": log}, ckpt)
        runs.commit()

    started, done_before = time.time(), sum(s["tokens"] for s in log["steps"])
    model.train()
    for epoch in range(step // per_epoch, epochs):
        if step >= max_steps:
            break
        order = data["train"][:]
        random.Random(42 + epoch).shuffle(order)
        for b in range(step - epoch * per_epoch, per_epoch):
            if step >= max_steps:
                break
            batch = order[b * accum:(b + 1) * accum]
            tokens = sum(sum(l != -100 for l in ex["labels"][1:]) for ex in batch)
            total = 0.0
            for ex in batch:                                   # token-weighted mean over the batch
                loss, _ = loss_of(ex)
                (loss / tokens).backward()
                total += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log["steps"].append({"step": step, "epoch": epoch + 1, "loss": total / tokens,
                                 "lr": schedule.get_last_lr()[0], "tokens": sum(len(ex["input_ids"]) for ex in batch)})
            rate = (sum(s["tokens"] for s in log["steps"]) - done_before) / (time.time() - started)
            print(f"step {step}/{steps} epoch {epoch + 1} loss {total / tokens:.4f} {rate:.0f} tok/s", flush=True)
            if step % CHECKPOINT_EVERY == 0 and step < max_steps:
                save_checkpoint()
        if step == (epoch + 1) * per_epoch or step >= max_steps:
            log["validation"].append({"step": step, "loss": evaluate()})
            print(f"validation loss after epoch {epoch + 1}:", log["validation"][-1]["loss"])
            save_checkpoint()
    log["peak_gpu_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    ckpt.unlink(missing_ok=True)
    runs.commit()
    return log


@app.local_entrypoint()
def main(epochs: int = 2, rank: int = 32, lr: float = 1e-4, accumulation: int = 8, smoke: bool = False):
    """modal run scripts/finetune/modal_app.py  (trains; the log lands in runs/finetune/)"""
    root = HERE.parents[1] / "runs" / "finetune"
    rows = {n: [json.loads(line) for line in open(root / f"{n}.jsonl")] for n in ("train", "validation")}
    hparams = {"epochs": epochs, "rank": rank, "lr": lr, "accumulation": accumulation}
    if smoke:     # the longest examples, so memory is measured at its worst
        rows = {n: sorted(r, key=lambda x: -len(json.dumps(x)))[:16 if n == "train" else 2] for n, r in rows.items()}
        hparams["max_steps"] = 2
    # spawned rather than awaited: the run carries on if this machine disconnects, and a rerun of
    # this command resumes from the last checkpoint
    call = train.spawn(rows["train"], rows["validation"], hparams)
    print("training call", call.object_id, flush=True)
    log = call.get()
    (root / ("smoke_log.json" if smoke else "train_log.json")).write_text(json.dumps(log, indent=1))
    print("validation loss:", [round(v["loss"], 4) for v in log["validation"]], "| peak GPU MiB:", log["peak_gpu_mib"])


@app.function(image=vllm_image, gpu="H200", volumes={"/weights": weights, "/runs": runs},
              secrets=[modal.Secret.from_dict(_api_key())], timeout=24 * 3600, scaledown_window=300)
@modal.concurrent(max_inputs=64)
@modal.web_server(8000, startup_timeout=1800)
def serve():
    adapter = Path("/runs") / RUN / "adapter"
    cmd = ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", "muse-glimmer",
           "--host", "0.0.0.0", "--port", "8000", "--api-key", os.environ["GLIMMER_API_KEY"],
           "--max-model-len", str(SERVE_LEN), "--gpu-memory-utilization", "0.92", "--max-num-seqs", "48",
           "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
           "--reasoning-parser", "muse_glimmer", "--generation-config", "auto"]
    if adapter.exists():
        # Text only: start-up otherwise profiles the vision encoder with a dummy image, and vLLM's LoRA
        # wrapping of the vision layers fails on patch inputs. The adapter leaves vision untouched.
        cmd += ["--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"glimmer-tuned={adapter}",
                "--limit-mm-per-prompt", json.dumps({"image": 0, "video": 0})]
    subprocess.Popen(cmd)
