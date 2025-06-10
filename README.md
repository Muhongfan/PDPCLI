# pdpcli

Piggy's data processing CLI (PDPCLI) utility
## Set up
`python experiments/evaluate.py` for starting the project.
## Guidance
* Generate feature engineering recommendations
   `pdpcli recommend -d example/energy_dataset.csv -t "price actual" --time time `
* Apply the recommendateions
   `pdpcli apply -d example/energy_dataset.csv -t "price actual" --time time --steps R1 R2 R3 R4 R5 R7 R9 -o output/output_energy.csv`
* Evaluate the recommendations with lgb
   `pdpcli evaluate --processed_file output/output_energy.csv -d example/energy_dataset.csv -t "price actual" --time time`

Energy dataset
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

[Electricity dataset](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting/data)
```sh
Loading data...
[1]     train's rmse: 170.857   valid's rmse: 172.656
Training until validation scores don't improve for 10 rounds
[2]     train's rmse: 154.804   valid's rmse: 162.029
......
[22]    train's rmse: 111.958   valid's rmse: 147.345
Early stopping, best iteration is:
[12]    train's rmse: 115.435   valid's rmse: 146.684

📊 Raw Data Evaluation:
   RMSE: 146.6840
   MAE : 117.4709
   R²  : 0.3779
   MAPE: 9.56%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
                       T2M_toc : 1.000
                    Holiday_ID : 0.868
                       T2M_dav : 0.755
                      QV2M_toc : 0.698
                       W2M_san : 0.585
                       TQL_dav : 0.472
                       TQL_toc : 0.434
                      QV2M_san : 0.396
                        school : 0.340
                       TQL_san : 0.321
                       W2M_toc : 0.283
                       T2M_san : 0.283
                      QV2M_dav : 0.189
                       W2M_dav : 0.170
[1]     train's rmse: 151.804   valid's rmse: 155.577
Training until validation scores don't improve for 10 rounds
[2]     train's rmse: 123.149   valid's rmse: 129.334
.....
[196]   train's rmse: 16.7785   valid's rmse: 29.8093
Early stopping, best iteration is:
[186]   train's rmse: 16.8541   valid's rmse: 29.8026

📊 Preprocessed Data Evaluation:
   RMSE: 29.8026
   MAE : 20.4603
   R²  : 0.9742
   MAPE: 1.71%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
                nat_demand_t-1 : 1.000
                          hour : 0.729
                       weekday : 0.500
                     dayofyear : 0.495
                      hour_sin : 0.458
                nat_demand_t-5 : 0.439
                nat_demand_t-2 : 0.425
                      hour_cos : 0.372
                nat_demand_t-4 : 0.322
                       T2M_toc : 0.321
                      QV2M_toc : 0.298
                       T2M_san : 0.290
                       W2M_san : 0.287
                       TQL_toc : 0.282
                       W2M_toc : 0.280
               nat_demand_lag1 : 0.269
                       TQL_san : 0.266
                       W2M_dav : 0.259
                       T2M_dav : 0.255
                nat_demand_t-3 : 0.251
                      QV2M_dav : 0.227
                      QV2M_san : 0.199
                       TQL_dav : 0.198
                           day : 0.196
                         month : 0.039
                        school : 0.033
```
[Solar Energy dataset](https://www.kaggle.com/datasets/chaitanyakumar12/time-series-forecasting-of-solar-energy/data)
```sh
Loading data...
/Users/amberm/Desktop/pdpcli/src/pdpcli/evaluate.py:177: DtypeWarning: Columns (2) have mixed types. Specify dtype option on import or set low_memory=False.
  df_raw = pd.read_csv(data_path, encoding="utf-8-sig")
[1]     train's rmse: 2378.12   valid's rmse: 2568.27
Training until validation scores don't improve for 10 rounds
[2]     train's rmse: 2230.05   valid's rmse: 2635.56
.....
[11]    train's rmse: 1542.52   valid's rmse: 3150.15
Early stopping, best iteration is:
[1]     train's rmse: 2378.12   valid's rmse: 2568.27

📊 Raw Data Evaluation:
   RMSE: 2568.2654
   MAE : 2147.6319
   R²  : -0.2368
   MAPE: 16.46%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
                    wind-speed : 1.000
                   temperature : 0.353
                wind-direction : 0.235
     average-pressure-(period) : 0.176
[1]     train's rmse: 1777.62   valid's rmse: 1706.21
Training until validation scores don't improve for 10 rounds
[2]     train's rmse: 1440.47   valid's rmse: 1383.06
[3]     train's rmse: 1172.02   valid's rmse: 1124.84
......
[199]   train's rmse: 111.173   valid's rmse: 112.101
[200]   train's rmse: 111.027   valid's rmse: 111.963
Did not meet early stopping. Best iteration is:
[200]   train's rmse: 111.027   valid's rmse: 111.963

📊 Preprocessed Data Evaluation:
   RMSE: 111.9628
   MAE : 80.7885
   R²  : 0.9972
   MAPE: 0.77%
[WARN] Model does not have evals_result_.
🧠 Normalized Top Features:
               solar_mw_smooth : 1.000
                  solar_mw_t-2 : 0.612
                 solar_mw_lag1 : 0.456
                  solar_mw_t-1 : 0.373
                  solar_mw_t-5 : 0.274
                  solar_mw_t-3 : 0.196
                  solar_mw_t-4 : 0.097
     average-pressure-(period) : 0.022
                    wind-speed : 0.021
                   temperature : 0.018
                wind-direction : 0.010
   average-wind-speed-(period) : 0.006
                      humidity : 0.006
```