# Tg ML Results

This folder collects the Tg prediction inputs and results for three polymer representations and three regression models.

## Representations

- `MorganFP`: pSMILES Morgan fingerprint, 2048 dimensions.
- `p-ECFP`: polymer extended-connectivity fingerprint, 2048 dimensions. The original result files use the token `ECFP`.
- `BigFP`: concatenated `p-ECFP` plus 40-dimensional topology fingerprint, 2088 dimensions.

## Models

- `SVR`: support vector regression.
- `GP`: Gaussian process regression.
- `RF`: random forest regression.

## Folder Layout

```text
ml_results/tg/
├── README.md
├── tg_model_summary.csv
├── features/
│   ├── MorganFP_2048.npy
│   ├── pECFP_2048.npy
│   ├── pTFP_topology_40.npy
│   └── BigFP_pECFP2048_pTFP40.npy
├── labels_and_splits/
│   ├── tg_exp.npy
│   └── tg_315_indices_folder1.npy ... tg_315_indices_folder5.npy
├── metrics/
│   ├── r2_score_tg_{train,test}_{svr,gp,rf}_{psmiles_MorganFP,ECFP,BigFP}.npy
│   └── RMSE_tg_{train,test}_{svr,gp,rf}_{psmiles_MorganFP,ECFP,BigFP}.npy
├── prediction_plots/
│   ├── by_model/
│   └── aggregate/
└── scripts/
    ├── final_prediction_tg.py
    ├── final_prediction_tg_loop.py
    └── final_prediction_tg_five_fold.py
```

## Summary Table

`tg_model_summary.csv` contains one row for each representation/model pair:

- train/test R2 mean and standard deviation
- train/test RMSE mean and standard deviation
- number of repeated runs per metric array

## Data Shapes

- `MorganFP_2048.npy`: `(315, 2048)`
- `pECFP_2048.npy`: `(315, 2048)`
- `pTFP_topology_40.npy`: `(315, 40)`
- `BigFP_pECFP2048_pTFP40.npy`: `(315, 2088)`
- `tg_exp.npy`: `(315,)`
- each fold index file: `(63,)`

## Notes

The per-model prediction plots are stored as PNG and SVG files in `prediction_plots/by_model/`. The SVG files include an explicit white background so axes and labels remain readable in dark-mode viewers.
The aggregate plots compare models for each representation and are stored in `prediction_plots/aggregate/`.
