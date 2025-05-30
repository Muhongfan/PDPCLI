# pdpcli

Piggy's data processing CLI (PDPCLI) utility

`python experiments/evaluate.py
`

```sh
🔍 Loading data...
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000218 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 990
[LightGBM] [Info] Number of data points in the train set: 7008, number of used features: 6
[LightGBM] [Info] Start training from score 837.185002

📊 Raw Data Evaluation:
   RMSE: 259.9068
   MAE : 75.5702
   R²  : 0.2993
   MAPE: inf%
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000843 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 2554
[LightGBM] [Info] Number of data points in the train set: 4336, number of used features: 16
[LightGBM] [Info] Start training from score 50.982751

📊 Preprocessed Data Evaluation:
   RMSE: 13.0887
   MAE : 2.8386
   R²  : 0.9765
   MAPE: inf%
```