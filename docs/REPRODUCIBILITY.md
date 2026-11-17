# 重現指南

## 範圍與必要材料

本 repo 提供自建 source/config、版本 pins 與彙總結果。模型、原始資料、adapter 和課程 `hw3/` 需另備；因此剛 clone 後可跑 CPU 結構測試，不能直接開始完整官方評測。

原實驗環境為 Ubuntu 24.04 / WSL2、Python 3.11.15、RTX 4090 24 GB。記錄的核心套件是 torch 2.11.0、transformers 5.13.1、peft 0.19.1、vllm 0.25.1、jinja2 3.1.6。這些是當時的固定環境，不表示最新版本；完整 pins 在 `code/env/requirements.lock.txt`。

以下指令在 Linux 專案根目錄執行。請使用足夠大的 Linux 資料磁碟；下載 Base、tokenizer、judge 和多來源資料會占用大量空間。不要在尚有重要成果的目錄盲目重跑資料生成。

## 1. 環境

先備好 NVIDIA 驅動、CUDA 可用環境與 [uv](https://docs.astral.sh/uv/)。安裝 C/C++ 編譯器後，建立本地環境：

```bash
sudo apt-get install -y gcc g++
uv venv .venv --python 3.11.15
uv pip sync code/env/requirements.lock.txt --python .venv/bin/python
```

`code/env/run.sh` 現在依自身位置尋找專案根目錄，優先使用 `.venv/bin`。若有舊工作區的 `$HOME/env/hw3.env` 仍會讀取；可用 `HW3_ENV_FILE` 指向自己的設定檔。歷史 `setup_workspace.sh` 用於 Windows/WSL 分開存放資料，包含當時路徑，不是 fresh clone 的必要步驟。

原 WSL 使用：

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false
```

將課程官方 pack 原封不動放在 `hw3/`，應包含 `eval.py`、`eval/`、`data/public_test.jsonl.gz`、`data/public_test_negotiation.jsonl.gz`。此目錄已由 Git 排除，不在本 repo 重新分發。

```bash
bash code/env/run.sh python code/env/check_env.py
bash code/env/run.sh python code/env/download_models.py
```

`--skip-judge` 可只下載 Base 與 tokenizer；完整文字評判仍需要 Qwen3-8B。原環境的 judge 另使用 CUDA 編譯工具，參考 `code/env/install_nvcc.sh`，先閱讀再執行。

## 2. 第一階段資料

下列操作會下載並執行第三方 BFCL 工具。工具 commit 已固定，原始碼下載在忽略的 data_work/raw；請先檢閱其來源與授權。此處不是繞過任何執行權限的方式。

```bash
mkdir -p data_work/raw data_work/processed data_work/frozen
bash code/env/run.sh python code/data/fetch_sources.py
bash code/env/run.sh python code/data/convert_sources.py
bash code/env/run.sh python code/data/fetch_bfcl_tools.py

# 必須先有碰撞清單，BFCL builder 與家族 guard 都需要它。
bash code/env/run.sh python code/data/dedupe_against_public.py \
  --out data_work/processed/collisions_bfcl.json \
  data_work/raw/bfcl/BFCL_v3_*.json

bash code/env/run.sh python code/data/build_bfcl_multiturn.py
bash code/env/run.sh bash code/data/run_data_v1.sh
bash code/env/run.sh bash code/data/run_data_v2.sh
bash code/env/run.sh bash code/data/run_data_v3.sh
bash code/env/run.sh bash code/data/run_data_v5.sh
```

data_v5 會保留 v1 的 Part 1、v2 的買家，再加入家族隔離及標籤修正後的 BFCL。這裡的 data_v2/data_v3 是早期混合資料版本，與後來的 gpt_policy_v2/v3 名稱不同。

## 3. E23 第一階段

```bash
bash code/env/run.sh python code/train.py --config configs/e23.json --run runs/E23
```

新 run 從 Base 開始；`--resume` 用於接回同一 run，會還原 optimizer/scheduler/抽樣狀態。`--init-adapter` 則只從父模型載入 adapter 權重，建立新的訓練行程，兩者不可混稱。

## 4. E29 接續資料與訓練

資料生成採獨占建立的地方，重跑請換新輸出路徑。以下應在新的資料工作區使用：

```bash
mkdir -p data_work/e27 data_work/e29
bash code/env/run.sh python code/data/gen_p1_verify_e27.py \
  --split train --seed 27101 --conversations 1500 --out data_work/e27/verify_train.jsonl
bash code/env/run.sh python code/data/dedupe_against_public.py --offer-frame-as-format \
  --out data_work/e27/collisions_public.json data_work/e27/verify_train.jsonl
bash code/env/run.sh python code/data/drop_hit_families.py \
  data_work/e27/collisions_public.json data_work/e27/verify_train.jsonl data_work/e27/verify_train_isolated.jsonl

bash code/env/run.sh python code/codex_p1_next_20261109/build.py --out data_work/cx_v1
bash code/env/run.sh python code/codex_p1_next_20261109/rewrite_natural.py \
  --source data_work/cx_v1 --out data_work/cx_natural

bash code/env/run.sh python code/ops/e27_build.py pool \
  --v5 data_work/frozen/data_v5/train.jsonl --verify data_work/e27/verify_train_isolated.jsonl \
  --out data_work/e29/base_pool.jsonl
bash code/env/run.sh python code/ops/e29_build.py pool \
  --base data_work/e29/base_pool.jsonl --codex data_work/cx_natural/train.jsonl \
  --v5 data_work/frozen/data_v5/train.jsonl --out data_work/e29/train_pool.jsonl
bash code/env/run.sh python code/ops/e29_build.py schedule \
  --pool data_work/e29/train_pool.jsonl --updates 250 --out-dir data_work/e29
bash code/env/run.sh python code/train.py --config configs/e29.json --run runs/E29 \
  --init-adapter runs/E23/checkpoints/step_1475
```

正式使用前，仍須對新生成的 cx_natural 做 QA 與 public 隔離，不以檔名相同當作驗證。既有產出的 natural_v2r1 train SHA-256 是 `be03b9811173f796cd745626f04f0ddedf2d89caf4c6b8d450e9e4d363bdf45d`。

## 5. 後續長政策資料

這些教材不屬於 E29 的同一批資料。v1 提供世界模型，v2/v3 使用個別撰寫的政策。輸出目錄已存在時，依程式的獨占建立／manifest 續做機制處理，不刪舊資料。

```bash
bash code/env/run.sh python code/gpt_policy_v1/generate.py
bash code/env/run.sh python code/gpt_policy_v1/audit.py
bash code/env/run.sh python code/gpt_policy_v1/build_v2.py
bash code/env/run.sh python code/gpt_policy_v1/build_v3.py
```

當晚的時間限制已改為選用 `--deadline 2026-11-17T23:00:00+08:00` 這種明確時間戳；預設不在每天固定時刻無聲停止。這不改變資料亂數或政策正文。生出 raw 後，每版仍須依序執行原 QA、去重、整家族排除、最後稽核，才有正式 train/dev：

```bash
# 對 v2 或 v3 擇一；先確保對應 raw 已完成。
D=data_work/processed/gpt_policy_v3
bash code/env/run.sh python code/data/qa_conversations.py --tokens "$D/train_raw.jsonl" "$D/dev_raw.jsonl"
bash code/env/run.sh python code/data/dedupe_against_public.py --out "$D/collisions.json" "$D/train_raw.jsonl" "$D/dev_raw.jsonl"
bash code/env/run.sh python code/data/drop_hit_families.py "$D/collisions.json" "$D/train_raw.jsonl" "$D/train.jsonl"
bash code/env/run.sh python code/data/drop_hit_families.py "$D/collisions.json" "$D/dev_raw.jsonl" "$D/dev.jsonl"
bash code/env/run.sh python code/gpt_policy_v1/final_audit_v3.py
```

## 6. 評估與測試

```bash
# 需要官方 pack 與對應 adapter；不與其他 GPU 工作同時執行。
bash code/env/run.sh bash code/eval_full.sh runs/eval_e29 runs/E29/checkpoints/step_250

# CPU-only，無模型／官方 pack 亦可執行。
python -B -m unittest discover -s code/audit -p test_check_predictions.py -v
python -B -m unittest discover -s code/gpt_policy_v1 -p test_time_budget.py -v

# 以下另需官方 pack 的 reader/scorer，但不需 GPU。
python -B -m unittest discover -s code/codex_eval_20261109 -p test_audit_part1.py -v
```

完整 GPU 訓練與 judge 未在本次 GitHub 整理中重跑；重現指令是依程式依賴整理，不宣稱已從空白機器完整重建。GPU 非決定性、批次與 runner 差異也可能造成小幅變動。舊 controller 內的固定時程、session 名稱及磁碟限制屬當時實驗設計，閱讀／建立新 session 後才使用，不作通用一鍵部署入口。
