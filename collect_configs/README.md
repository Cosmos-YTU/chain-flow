# Collection Configs

Teacher-state collection configs are kept flat because existing tests and scripts reference these names directly.

## Smoke/local

- `smoke_gsm8k.yaml`: small offline smoke collection.
- `smoke_gsm8k_online.yaml`: small online smoke collection.

## GSM8K train sets

- `64_train_gsm8k.yaml` and `64_train_gsm8k_resume.yaml`
- `1k_train_gsm8k.yaml`
- `3k_train_gsm8k.yaml`
- `3.5k_train_gsm8k.yaml` and `3.5k_train_gsm8k_resume.yaml`

## GSM8K test sets

- `1k_test_gsm8k.yaml`

Prefer adding new collection configs with explicit dataset size and split in the filename.
