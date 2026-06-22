"""guido_sft.py — esempio MINIMALE di SFT (response-masked) di Guido nel PiCOFormer di produzione.

Idea: loss SOLO sulla risposta. Si tokenizza prompt e response separatamente e si mette il
target a -100 sui token del prompt → `pico_cross_entropy_loss` (cce, ignore_index=-100) li ignora.
Template per il team: sostituire DATA con il vero SubCorpus QA e Muon come optimizer.

Run (1 GPU): python examples/guido_sft.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from Vathos.guido import load_guido
from Vathos.picoformer import configure_picoformer_training_runtime, pico_cross_entropy_loss

CKPT = os.environ.get("CKPT", "/leonardo_work/IscrC_YENDRI/paerle/PiCO/ckpts/guido_v4/step_138000.pt")
SEQ = 1024
IGNORE = -100

# (prompt, response) — il prompt va MASCHERATO; il modello impara a produrre solo la response.
# Template eval-aware: "### MATH\n\nProblem: …\nSolution: " → "{soluzione}".
DATA = [
    ("### MATH\n\nProblem: What is 7 multiplied by 6?\nSolution: ",
     "7 multiplied by 6 is 42.\nThe answer is \\boxed{42}."),
    ("### MATH\n\nProblem: Sum the first n even numbers.\nSolution: ",
     "The sum of the first n even numbers 2+4+...+2n is n(n+1). \\boxed{n(n+1)}"),
]


def make_batch(tok, pairs, device):
    """Costruisce (x, y) con y=-100 sui token del prompt (loss solo sulla response + EOS)."""
    eos = tok.eos_token_id
    xs, ys = [], []
    for prompt, resp in pairs:
        p = tok.encode(prompt, add_special_tokens=False)
        r = tok.encode(resp, add_special_tokens=False) + [eos]
        seq = (p + r)[:SEQ]
        mask = ([0] * len(p) + [1] * len(r))[:SEQ]           # 1 = response
        if len(seq) < SEQ:                                    # pad (target -100 sul pad)
            seq = seq + [eos] * (SEQ - len(seq)); mask = mask + [0] * (SEQ - len(mask))
        t = torch.tensor(seq)
        y = t.clone()
        y[torch.tensor(mask) == 0] = IGNORE                   # maschera prompt+pad
        xs.append(t[:-1]); ys.append(y[1:])                   # next-token shift
    return torch.stack(xs).to(device), torch.stack(ys).to(device)


def main():
    configure_picoformer_training_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_guido(CKPT, device=device, dtype=torch.bfloat16, for_inference=False)
    model.train()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1", use_fast=True)

    x, y = make_batch(tok, DATA, device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.95))  # demo; in reale: Muon
    print(f"[sft] esempi={x.size(0)} seq={x.size(1)}  loss-solo-su-response (-100 sul prompt)")
    for step in range(20):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = pico_cross_entropy_loss(model, x, y, softcap=cfg.logit_softcap)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 5 == 0:
            print(f"  step {step:2d}  loss {loss.item():.4f}", flush=True)
    print("[sft] OK — loop SFT response-masked funzionante (sostituire DATA col vero SubCorpus QA).")


if __name__ == "__main__":
    main()
