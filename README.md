# pdpcli

Piggy's data processing CLI (PDPCLI) utility
This Python package aims to compare the performance impact of feature engineering pipelines under different optimized execution orders while keeping the feature set consistent.
## Set up
`python experiments/evaluate.py` for starting the project.
## Guidance
* Generate feature engineering recommendations
   `pdpcli recommend -d example/energy_dataset.csv -t "price actual" --time time `
* Apply the recommendateions
   `pdpcli apply -d example/energy_dataset.csv -t "price actual" --time time --steps R1 R2 R3 R4 R5 R7 R9 -o output/output_energy.csv`
* Evaluate the recommendations with lgb
   `pdpcli evaluate --processed_file output/output_energy.csv -d example/energy_dataset.csv -t "price actual" --time time`

The evaluated results:
```sh
🔍 Loading data...
[1]     train's rmse: 12.6691   valid's rmse: 13.2662
Training until validation scores don't improve for 10 rounds
.....
[29]    train's rmse: 7.44536   valid's rmse: 10.2341
Early stopping, best iteration is:
[19]    train's rmse: 7.8633    valid's rmse: 10.1235

📊 Raw Data Evaluation:
   RMSE: 10.1235
   MAE : 7.8211
   R²  : 0.2362
   MAPE: 13.04%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
               price day ahead : 1.000
            generation nuclear : 0.925
generation hydro run-of-river and poundage : 0.562
            generation biomass : 0.550
   generation fossil hard coal : 0.512
              generation waste : 0.512
    generation other renewable : 0.463
         generation fossil oil : 0.425
              generation other : 0.338
generation fossil brown coal/lignite : 0.312
             total load actual : 0.275
         generation fossil gas : 0.225
      forecast solar day ahead : 0.212
generation hydro water reservoir : 0.163
           total load forecast : 0.163
forecast wind onshore day ahead : 0.150
       generation wind onshore : 0.150
              generation solar : 0.138
generation hydro pumped storage consumption : 0.050
[1]     train's rmse: 9.35879   valid's rmse: 10.6572
Training until validation scores don't improve for 10 rounds
.....
[200]   train's rmse: 1.12645   valid's rmse: 1.34959
Did not meet early stopping. Best iteration is:
[200]   train's rmse: 1.12645   valid's rmse: 1.34959

📊 Preprocessed Data Evaluation:
   RMSE: 1.3496
   MAE : 0.9767
   R²  : 0.9738
   MAPE: 1.45%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
           price actual_smooth : 1.000
              price actual_t-2 : 0.715
             price actual_lag1 : 0.657
              price actual_t-1 : 0.619
               price day ahead : 0.558
                     dayofyear : 0.396
                          hour : 0.322
                           day : 0.240
                      hour_sin : 0.220
                      hour_cos : 0.217
         generation fossil gas : 0.188
              price actual_t-5 : 0.171
              generation solar : 0.162
              price actual_t-3 : 0.160
generation hydro pumped storage consumption : 0.159
              price actual_t-4 : 0.157
   generation fossil hard coal : 0.143
generation hydro water reservoir : 0.133
           total load forecast : 0.133
         generation fossil oil : 0.132
              generation waste : 0.124
                       weekday : 0.106
       generation wind onshore : 0.105
generation hydro run-of-river and poundage : 0.104
            generation biomass : 0.100
forecast wind onshore day ahead : 0.096
    generation other renewable : 0.092
      forecast solar day ahead : 0.091
              generation other : 0.088
            generation nuclear : 0.086
             total load actual : 0.083
generation fossil brown coal/lignite : 0.060
                         month : 0.055
```
