# Grand Challenge

## TODO

| 可以試的事                                  | 結果                                       |
| -------------------------------------- | ---------------------------------------- |
| 只用 `GRU` ，不用 `Dense`                   | 結果有比較好一點                                 |
| 把 `loss` 改成 `mse` 看看                   | 結果有比較好一點，而且 `vali` 的 `loss` 比較不會overfit 之後就快速下降 |
| 試試看不要只 append 3 個句子，多 append  2 個和 1 個 | 來囉                                       |
| 把 `GRU` 的值改小一點，避免太快 `overfitting`      | `GRU(128)` 還不錯<br>`GRU(64)` 也差不多…<br>數字大小真的有差嗎 = = |
| `GRU` 裡面不要 `dropout` ，避免句子文法被斷掉？       |                                          |



## 分數紀錄

| ID   | vali score | kaggle score | description                              | machine  | comment |
| ---- | ---------- | ------------ | ---------------------------------------- | -------- | ------- |
| 1    | `0.68366`  |              | `RNN(dropout(0.3))`<br>`text_data`: 0 ~ 1/5 | `aiuser` |         |
| 2    | `0.67983`  | `0.38800`    | `RNN(dropout(0.5))`<br>`text_data`: all  | `aiuser` |         |
|      |            |              |                                          |          |         |

