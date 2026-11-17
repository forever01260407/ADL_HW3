# 資料與來源

## 第一階段 data_v5

| 來源 | 對話 | 原始 assistant 訊息 | 排除 no_target 後的有效目標 |
|---|---:|---:|---:|
| ToolACE | 2,197 | 2,600 | 2,600 |
| Glaive function calling v2 | 848 | 2,800 | 2,800 |
| Hermes function calling v1 | 302 | 600 | 600 |
| 合成工具鏈 | 1,171 | 8,002 | 8,002 |
| 合成買家 | 1,774 | 8,001 | 8,001 |
| BFCL plain | 83 | 830 | 753 |
| BFCL state | 98 | 917 | 853 |
| **合計** | **6,473** | **23,750** | **23,609** |

有效目標數、長度篩選後的可用數和訓練 schedule 抽樣數是不同概念。BFCL 的 141 則 no_target 訊息保留於歷史，不當作答案計算 loss。

BFCL 工具本體從上游下載至本地 data_work，不放進這個 repository。參數可見性、初始狀態與變體家族處理在 `prepare_bfcl_v5.py`、`bfcl_family_guard.py`。substring / 型別檢查只是部分 QA，不能視為完整語意審核。

## E29 的接續教材與實際抽樣

E29 從 E23 接續 250 次更新，共 4,000 個位置，依固定 seed 分桶抽樣、不重複：

| 類型 | 位置 |
|---|---:|
| data_v5 買家 | 1,356 |
| natural_v2r1 工具後呼叫 | 600 |
| natural_v2r1 工具後文字 | 200 |
| 第一步文字確認 | 600 |
| E27 先查再做教材 | 400 |
| 其他 data_v5 Part 1 | 844 |

`natural_v2r1` 是 2,144 段、5,392 個有效目標；新增干擾工具由 `e29_build.py` 處理。它與後來命名為 `gpt_policy_v2` 的教材是兩件事，請勿混淆。

## 後續長政策資料

| 版本 | 政策／領域 | train | dev | 有效 train / dev 目標 |
|---|---:|---:|---:|---:|
| gpt_policy_v2 | 20 | 439 | 78 | 2,350 / 384 |
| gpt_policy_v3 | 20 個全新領域 | 439 | 78 | 2,350 / 384 |

政策是由 LLM 逐篇撰寫的完整字串，對話由程式世界展開；不是每段都另由教師模型自由生成。每版保留三個完整領域作 dev，仍共享部分流程程式，不能當作完全獨立的未知流程測試。

這些版本沿用同一套世界邏輯，因此不要把相應版本／家族變體分散至 train 和 dev。v1 有共用政策模板的限制，保留作歷史比較，不作為獨立自然語言政策成果。

## 隔離與品質檢查

- `qa_conversations.py`：工具格式、參數型別、資訊追溯、數值與模板／長度檢查。
- `dedupe_against_public.py`：正規化文字、五詞片段重疊；是字面檢查，不是語意去重證明。
- `drop_hit_families.py`：任何變體有命中時排除整個 family。
- `no_target`：必須透過 `common.expand_steps` 尊重排除索引。
- 合成世界另以可見歷史檢查和錯誤參數突變測試檢查標籤；無法取代全部人工審閱。

公開測試用於理解任務型態、錯誤診斷、隔離比對及最後版本取捨；沒有將公開測試對話作為訓練樣本。買家固定介面／格式由課程資料抽取，不能把這件事隱瞞成完全未接觸任何公開格式。

## 上游來源與版本

- [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
- [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla)：BFCL 工具執行器。

確切 revision / commit 在 [data_revisions.json](../records/data_revisions.json)。模型 pins 在 [model_revisions.json](../records/model_revisions.json)。模型權重、原資料與第三方工具原始碼均不隨本 repo 重新散布。
