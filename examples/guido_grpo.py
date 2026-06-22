"""guido_grpo.py — esempio MINIMALE di step GRPO con Guido nel PiCOFormer di produzione.

Pipeline GRPO (un prompt → gruppo di G rollout → reward → advantage group-relative → policy loss):
  1. generate_grpo_rollouts: campiona G completamenti dallo stesso prompt (KV-cache statica).
  2. reward(completamento): scalare per rollout (QUI: toy reward, sostituire col vero verificatore).
  3. advantage = (reward - media_gruppo) / std_gruppo   (GRPO, baseline = media del gruppo).
  4. completion_logprobs(policy): logprob per-token gradabili; (opz.) ref policy in no_grad per KL.
  5. loss = -(advantage * logprob_completamento).mean()  [+ beta * KL(policy || ref)].

Template per il team: usare `compile_decode=True` per velocità, group_size grande, e un reward reale.
Run (1 GPU): python examples/guido_grpo.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from Vathos.guido import load_guido
from Vathos.picoformer import DecodeConfig, generate_grpo_rollouts, completion_logprobs

CKPT = os.environ.get("CKPT", "/leonardo_work/IscrC_YENDRI/paerle/PiCO/ckpts/guido_v4/step_138000.pt")
GROUP = int(os.environ.get("GROUP", 8))


def toy_reward(text: str) -> float:
    """Reward FINTO (placeholder): premia la presenza di \\boxed{...}. Sostituire col verificatore vero
    (es. math_verify sulla risposta boxed vs gold)."""
    return 1.0 if "\\boxed{" in text else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_guido(CKPT, device=device, dtype=torch.bfloat16, for_inference=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mistralai/Mathstral-7B-v0.1", use_fast=True)

    prompt = "### MATH\n\nProblem: Compute 7 multiplied by 6.\nSolution: "
    ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=device)

    # 1. rollout di gruppo (compile_decode=True consigliato per velocità)
    decode = DecodeConfig(max_new_tokens=128, group_size=GROUP, temperature=1.0,
                          eos_token_id=tok.eos_token_id, return_logprobs=True, compile_decode=False)
    batch = generate_grpo_rollouts(model, ids, decode)

    # 2. reward per rollout
    comp = batch.sequences[:, batch.prompt_len:]
    texts = [tok.decode(row.tolist(), skip_special_tokens=True) for row in comp]
    rewards = torch.tensor([toy_reward(t) for t in texts], device=device, dtype=torch.float32)

    # 3. advantage GRPO (baseline = media del gruppo)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # 4. logprob della policy (gradabili) allineati ai token di completamento
    logp = completion_logprobs(model, batch)              # [G, T], 0 dopo EOS via completion_mask
    seq_logp = logp.sum(dim=-1)                            # logprob del completamento per rollout

    # 5. policy loss GRPO (qui senza KL; aggiungere beta*KL(policy||ref) con una ref in no_grad)
    loss = -(adv.detach() * seq_logp).mean()

    print(f"[grpo] group={GROUP}  rewards={rewards.tolist()}")
    print(f"[grpo] advantages={[round(a,2) for a in adv.tolist()]}")
    print(f"[grpo] policy loss={loss.item():.4f}  (gradabile → loss.backward() + opt.step())")
    print("[grpo] OK — rollout + reward + advantage + policy-logprob funzionanti. "
          "Sostituire toy_reward col verificatore vero e chiudere il loop RL.")


if __name__ == "__main__":
    main()
