# Grand Challenge

## TODOs

| 可以試的事                                  | 結果                                       |
| -------------------------------------- | ---------------------------------------- |
| `GRU` 裡面不要 `dropout` ，避免句子文法被斷掉？       | 在 `daniel` 上面有滿好的效果（`0.69879`)<br>(`GRU(128)`) |
| `batch_size` 改為 64                     | 在 `daniel` 上面有滿好的效果（`0.69879`)<br>(`GRU(128)`) |
| 換另一種 `embedding` 看兩個的結果有沒有差            |                                          |
| 不切 `validation set`                    |                                          |
| 試試看不要只 append 3 個句子，多 append  2 個和 1 個 |                                          |

## Finished TODOs

| apply or not | 完成的事                                     | 結果                                       |
| ------------ | ---------------------------------------- | ---------------------------------------- |
| Y            | 只用 `GRU` ，不用 `Dense`                     | 結果有比較好一點                                 |
| Y            | 把 `loss` 改成 `mse` 看看                     | 結果有比較好一點，而且 `vali` 的 `loss` 比較不會overfit 之後就快速下降 |
| Y            | 把 `GRU` 的值改小一點，避免太快 `overfitting`        | `GRU(128)` 還不錯<br>`GRU(64)` 也差不多…<br>數字大小真的有差嗎 = = |
| Y            | `GRU` 改用預設的 `activation function` (`tanh`) 看看 | `0.69202`                                |
| Y            | `GRU` 的數字改成不是 400 的因數或倍數                 | `0.69202`                                |
| N            | 加一層 `dense`                              | 結果都比較爛一點，還是不要加好了                         |

## 分數紀錄

|  ID  | vali score | kaggle score | description                              | machine  | comment                                  |
| :--: | ---------- | ------------ | ---------------------------------------- | -------- | ---------------------------------------- |
|  1   | `0.68366`  |              | `RNN(dropout(0.3))`<br>`text_data`: 0 ~ 1/5 | `aiuser` |                                          |
|  2   | `0.67983`  | `0.38800`    | `RNN(dropout(0.5))`<br>`text_data`: all  | `aiuser` |                                          |
|  3   | `0.68972`  | `0.48400`    | `RNN(dropout(0.5))`<br>`loss` : `mse`    | `aiuser` | 跟上一個也差太多 = =<br>看來 `mse` 比 `binary_crossentropy` 好很多！ |
|  3   | `0.67368`  | `0.41200`    | concate **2** together<br>`GRU(128, dropout(0.4))`<br>`q_maxlen` = 132<br>`a_maxlen` = 72 | `daniel` | 看來還是 concate 3 個的結果比較好！                  |
|  5   | `0.69014`  | `0.46000`    | fix bug (not generate all answers)       | `azure`  | 看來結果沒差太多，問題不是在這裡                         |
|  6   | `0.69790`  | `0.52800`    | normalize (before cosine similarity)<br>batch_size(1024) | `azure`  | 結果是有比較好，不過還不是很OK…                        |
|  8   | `0.72359`  | `0.58799`    | fix `cos_similarity` formula<br>no dropout<br>no dense | `azure`  | 結果還真的變好了…                                |
|  9   | `0.71222`  | `0.56000`    | `GRU(128)`<br>`dropout(0.5)`<br>`dense(32, 'relu')` | `aiuser` | 結果比較爛<br>可見 `dense` 的效果並不好               |
|  10  | `0.72535`  | `0.56799`    | `GRU(256)`                               | `azure`  | 看來還是 `128` 比較好？                          |

## 加總的分數紀錄

| ID       | vali score | description          | comment                        |
| -------- | ---------- | -------------------- | ------------------------------ |
| 9\_10    | `0.58800`  | `9` + `10`           | 分數有比 `9` 和 `10` 本身要好了 2 ~ 3 了！ |
| 8\_9\_10 | `0.61999`  | 2 * `8` + `9` + `10` | 分數大突破！！！                       |

