import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "finetune"))
from sft_tokens import trained_spans  # noqa: E402


def test_only_the_models_own_turns_are_trained():
    text = ("<|start|>system<|message|>sys<|eot|><|start|>user<|message|>go<|eot|>"
            "<|start|>assistant to=self<|message|>think<|eom|>"
            "<|start|>assistant to=get_logs<|message|>CALL<|eot|>"
            "<|start|>tool get_logs<|message|>OUT<|eot|>"
            "<|start|>assistant to=submit_ranking<|message|>DONE<|eot|>")
    got = [text[a:b] for a, b in trained_spans(text)]
    assert got == [" to=self<|message|>think<|eom|>",                       # after the prompt's header
                   "<|start|>assistant to=get_logs<|message|>CALL<|eot|>",  # the model wrote this header
                   " to=submit_ranking<|message|>DONE<|eot|>"]              # new turn after a tool result
