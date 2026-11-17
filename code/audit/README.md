# 獨立評測輸出檢查

`check_predictions.py` 只使用 Python 標準函式庫，不載入模型、不使用 GPU、不修改預測或官方程式。不輸出題目、對話、答案或個別 ID；只保存統計與檔案摘要。這是結構檢查，不是官方分數，也不能用來判定訓練資料合規或模型身分。

在 Linux 專案根目錄、已啟用環境時執行：

```bash
# 生成中：允許尚未完成的題目，暫緩檢查最後一行未寫完的資料。
python -B code/audit/check_predictions.py --data hw3/data/public_test.jsonl.gz --pred runs/E02/predictions/tools.jsonl --in-progress --out records/audits/e02_tools_progress_NEW.json

# 正式完成後：不可加 --in-progress；缺題會失敗。
python -B code/audit/check_predictions.py --data hw3/data/public_test.jsonl.gz --pred runs/E02/predictions/tools.jsonl --out records/audits/e02_tools_final_NEW.json

# Part 2 預測完成後另查。
python -B code/audit/check_predictions.py --data hw3/data/public_test_negotiation.jsonl.gz --pred runs/E02/predictions/negotiation.jsonl --out records/audits/e02_negotiation_final_NEW.json

# 只使用自建資料的驗證。
python -B -m unittest discover -s code/audit -p test_check_predictions.py -v
```

每次保存報告須換成新的 `--out` 路徑；程式拒絕覆寫既有檔案。也可省略 `--out`，只看終端結果。Windows 原生 Python 可執行這個工具，但 WSL 內的預測路徑須從 WSL 呼叫。

狀態：`pass` 代表這次讀到的所有預期 ID 都有結構合法的結果；`pending` 代表生成中尚有缺題或未完成的尾行；`fail` 代表發現錯誤。只有 `fail` 回傳 exit code 1；自動化要宣告完成時必須使用不含 `--in-progress` 的命令。

截斷（`length`）和超長輸入（`overlength`）是官方輸出的合法狀態，會列入統計而不作為結構錯誤；官方評分仍可把它們判錯。本工具不過濾、不修補任何結果。

進行中的檢查只取開檔時所見大小的快照，假設原程序只往後追加。它不鎖檔，亦不處理另一程序同時替換或改寫預測的情形。`snapshot_sha256` 是此次讀取片段的摘要，不是最終完整檔案摘要。

續跑前模型／設定 manifest 的核對仍是另一項工作；本工具不提供該保證。
