# Data Manifest

This directory intentionally contains no binary shards. `train_A100.py` reads the
existing Guido packed-token shard format directly:

```text
<corpus_root>/<dataset>/
  index.json
  shard_00000.bin
  shard_00001.bin
  ...
```

Known Leonardo datasets:

```text
/leonardo_scratch/fast/IscrC_YENDRI/mprignan/corpus_v2/
  fineweb_edu
  cosmopedia
  openmath_full
  openmathreasoning
  tinygsm
  numina_15
  numina_cot

/leonardo_scratch/large/userexternal/mprignan/corpus_v4/
  fineweb_edu_v4
  finemath_3plus
  nemotron_cc_math_4plus
```

Useful presets in `train_A100.py`:

```bash
--mixture-preset default
--mixture-preset v3_noloops
--mixture-preset fineweb_only
--mixture-preset fineweb_v4_only
```

For a custom mix:

```bash
python train_A100.py --mixture "fineweb_edu=0.25,cosmopedia=0.05,openmath_full=0.70"
```

The trainer infers `vocab_size`, `eos_id`, and dtype from each dataset's
`index.json` and fails loudly if a mixture combines incompatible tokenizers.
