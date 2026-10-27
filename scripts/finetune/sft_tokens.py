"""Tokenize chat examples with Glimmer's own template and mask everything but its turns.

Axolotl takes the result as a pre-tokenized dataset (input_ids, attention_mask,
labels). Tokenizing here rather than in Axolotl keeps tool-call arguments as
objects; a datasets table would merge every tool's arguments into one schema
and the template would render the other tools' parameters as null.

Masking follows how Glimmer generates. The prompt ends with `<|start|>assistant`,
so the model writes everything after it: the recipient (` to=self` for
reasoning, ` to=<tool>` for a call), the message, and the end token. After an
`<|eom|>` the turn goes on and the model writes the next `<|start|>assistant`
header itself, so that header is trained too.
"""

import re

BLOCK = re.compile(r"<\|start\|>(\w+)(.*?)(<\|eot\|>|<\|eom\|>)", re.S)
HEADER = len("<|start|>assistant")


def trained_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges the model is trained to produce."""
    spans, continuing = [], False
    for m in BLOCK.finditer(text):
        if m.group(1) == "assistant":
            spans.append((m.start() if continuing else m.start() + HEADER, m.end()))
            continuing = m.group(3) == "<|eom|>"
        else:
            continuing = False
    return spans


def tokenize(tokenizer, example: dict, max_len: int) -> dict | None:
    """{'input_ids', 'attention_mask', 'labels'}, or None if the example is longer than max_len."""
    text = tokenizer.apply_chat_template(example["messages"], tools=example.get("tools"), tokenize=False)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_len:
        return None
    spans, labels, j = trained_spans(text), [], 0
    for tok, (start, _) in zip(ids, enc["offset_mapping"]):
        while j < len(spans) and start >= spans[j][1]:
            j += 1
        labels.append(tok if j < len(spans) and spans[j][0] <= start else -100)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
