"""grpo_rollout.py — rollout GRPO del modello Guido nel PiCOFormer di produzione (nuovo Aplos).
Mostra la generazione interna su un problema, in due framing (MATH-wrapped e raw), e misura tok/s.

Run: python grpo_rollout.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

CKPT = os.environ.get("CKPT", "/leonardo_work/IscrC_YENDRI/paerle/PiCO/ckpts/guido_v4/step_138000.pt")
CHANNEL = os.environ.get("CHANNEL", "torch_leaky_reglu2")
GROUP = int(os.environ.get("GROUP", 6))
MAXNEW = int(os.environ.get("MAXNEW", 256))
TEMP = float(os.environ.get("TEMP", 0.8))
os.environ.setdefault("HF_HOME", "/leonardo_scratch/fast/IscrC_YENDRI/mprignan/.cache/huggingface")

from Vathos.picoformer import (PiCOFormerConfig, DecodeConfig, build_picoformer,
                               configure_runtime, generate_grpo_rollouts,
                               prepare_for_inference)

PROBLEM = ("Determine the closed form S(n) for the sum of every pair number up to n. "
           "ex: S(6) = 2 + 4 + 6")

def load_model(device, dtype):
    cfg = PiCOFormerConfig(
        vocab_size=32768, d_model=1280, n_layers=24, n_heads=20, n_kv_heads=20,
        max_seq_len=8192, rope_base=1_000_000.0, attention="gated_mha",
        channel=CHANNEL, ffn_multiplier=3, qk_norm=True,
        attention_gate_input_dim=128, logit_softcap=30.0, tied_embeddings=True,
        smear_gate=True, smear_gate_input_dim=128,
    )
    model = build_picoformer(cfg)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = {(k[9:] if k.startswith("backbone.") else k): v for k, v in raw["model"].items()}
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f"load impuro: missing={miss} unexpected={unexp}"
    return prepare_for_inference(model, device=device, dtype=dtype)


def rollout(model, tok, prompt_text, label, eos_id, device):
    ids = torch.tensor(tok.encode(prompt_text, add_special_tokens=False), device=device)
    decode = DecodeConfig(max_new_tokens=MAXNEW, group_size=GROUP, temperature=TEMP,
                          eos_token_id=eos_id, return_logprobs=True, compile_decode=False)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    batch = generate_grpo_rollouts(model, ids, decode)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0

    seqs = batch.sequences                     # [G, L_total]
    plen = ids.numel()
    comps = seqs[:, plen:]                      # [G, <=MAXNEW]
    # token generati reali (fino a EOS per ciascun rollout)
    gen_tok = 0
    rows = []
    for i in range(comps.size(0)):
        row = comps[i].tolist()
        if eos_id in row:
            row = row[:row.index(eos_id) + 1]
        gen_tok += len(row)
        rows.append(tok.decode(row, skip_special_tokens=True))
    tps = gen_tok / dt
    print("\n" + "=" * 78)
    print(f"### FRAMING: {label}")
    print(f"prompt ({ids.numel()} tok): {prompt_text!r}")
    print(f"GRPO: group={GROUP} max_new={MAXNEW} temp={TEMP}  →  {gen_tok} tok in {dt:.2f}s = {tps:,.0f} tok/s")
    print("=" * 78)
    for i, r in enumerate(rows):
        print(f"\n--- rollout {i+1}/{GROUP} ---\n{r}")
    return tps, gen_tok, dt


def main():
    configure_runtime()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[grpo] device={device} dtype={dtype} channel={CHANNEL}")
    model = load_model(device, dtype)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1", use_fast=True)
    eos = tok.eos_token_id

    # warmup (esclude setup/lazy-init dal timing)
    _ = generate_grpo_rollouts(model, torch.tensor(tok.encode("warmup", add_special_tokens=False), device=device),
                               DecodeConfig(max_new_tokens=8, group_size=2, temperature=1.0,
                                            eos_token_id=eos, return_logprobs=False, compile_decode=False))
    torch.cuda.synchronize()

    wrapped = f"### MATH\n\nProblem: {PROBLEM}\nSolution: "
    raw = PROBLEM
    r1 = rollout(model, tok, wrapped, "MATH-WRAPPED", eos, device)
    r2 = rollout(model, tok, raw, "RAW (nessun wrap)", eos, device)

    print("\n" + "#" * 78)
    print("# RIASSUNTO VELOCITÀ (GRPO, eager / compile_decode=False)")
    print(f"#   wrapped : {r1[0]:,.0f} tok/s  ({r1[1]} tok / {r1[2]:.2f}s)")
    print(f"#   raw     : {r2[0]:,.0f} tok/s  ({r2[1]} tok / {r2[2]:.2f}s)")
    print("#" * 78)

if __name__ == "__main__":
    main()
