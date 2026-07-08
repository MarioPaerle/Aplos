# RoPE fp32 fix — GRPO reward-decay root cause (2026-07-08)

**File patched:** `Vathos/_spatials.py` → `RoPE._update_cache` (base `RoPE` class, ~line 42)
**Backup of pre-fix file:** `Vathos/_spatials.py.bak_ropefix_20260708`
**Author:** debugging session with Mario (guido_v5 GRPO run).

## Symptom
GRPO fine-tuning of guido_v5 (0.5B, `torch_swiglu`, 24L d1280) slowly **lost** reward
instead of gaining it; boxed-answer accuracy bled downward over ~180 steps with no crash,
no NaN. SFT and `generate_grpo` eval were both strong — only GRPO regressed.

## Root cause
The model has two RoPE cos/sin table builders at different precision:

- **Scorer path** — teacher-forced `PiCOFormer.forward` → `RoPE.forward` → `_update_cache`,
  which built `t = torch.arange(seq_len, dtype=q.dtype)` with `q.dtype == bfloat16` (inference
  runs in pure bf16), and used a bf16-cast `inv_freq` (non-persistent buffer, cast by
  `model.to(bf16)`). → **bf16-quantized positions AND frequencies.**
- **Compiled sampler path** — `generate_grpo(compile_decode=1)` → `RoPE.prebuild` → uses
  `torch.arange(max_len, dtype=torch.float32)` and `inv_freq.float()`. → **exact fp32.**

bf16 cannot represent consecutive integers past ~256 (they snap to multiples of 2/4/8/16),
so the scorer rotated late-completion tokens by the wrong angles. The model that **generates**
(fp32 RoPE, strong) and the model that is **scored/optimized** (bf16 RoPE, corrupted) were
effectively different policies. GRPO computes its gradient from the scorer's logprobs, so
every update nudged a good policy in a slightly wrong direction → slow reward decay.

## Evidence
- Telemetry `samax = max|sampling_logprobs − scorer_logprobs|` at batch 1, temp 1.0
  (invariant `test_ratio_at_init_is_one` requires < 1e-4):
  - compiled decode: **samax ≈ 23.6**
  - eager decode:    **samax ≈ 1.86** (sampler & scorer share the same bf16 table → they
    agree *because they share the bug*, not because they're correct)
- CPU check, bf16-position vs fp32-position RoPE cos-table (base 1e6, dim 64, 600 pos):
  max abs diff **1.89** on a [-1,1] range (0.62 even at pos ≤ 256 from bf16 `inv_freq`).
- Two independent code audits confirmed this and **exonerated** `for_inference=True`
  (numerically inert: only `.eval()`, dropout=0) and the SwiGLU/softcap/norm/tying paths
  (identical across scorer and sampler). Architecture matches v5 training config field-for-field.

## The fix
`RoPE._update_cache` now builds the tables in fp32 and casts to the working dtype at the end,
mirroring `prebuild()`:

```python
t = torch.arange(seq_len, device=device, dtype=torch.float32)
freqs = torch.outer(t, self.inv_freq.to(device).float())
self._cos_cached = freqs.cos().to(dtype)
self._sin_cached = freqs.sin().to(dtype)
```

After this, scorer forward, eager sampler and compiled sampler all share the correct fp32
RoPE; `samax` should collapse toward the ~2-nat bf16 floor and the GRPO ratio-at-init
invariant should hold in every mode.

## Known latent twin (NOT patched — out of scope for guido_v5)
`YaRNScaledRoPE._update_cache` (~line 210) has the same `dtype=dtype` bf16 build. guido_v5
uses the base `RoPE`, so it is unaffected, but any model using the YaRN variant carries the
same bug. Apply the identical fp32 treatment there before using YaRN for RL scoring.

## Verify / revert
- Verify: re-run the compiled logprob diag (`slurm/kl_logprob_diag_cg_t512.slurm`); `samax`
  should drop from ~23.6 to ~2.
- Revert: `cp Vathos/_spatials.py.bak_ropefix_20260708 Vathos/_spatials.py`
