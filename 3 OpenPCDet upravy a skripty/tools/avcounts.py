from pathlib import Path
import pickle
import pandas as pd
import numpy as np

ROOT = Path("./2025-11-11_nested")
METHOD_NAMES = ["pv-3b_sp", "pv-5b_sp", "lai-5b_sp", "oi-5b", "oi-5b_sp"]

for m in METHOD_NAMES:
    for t in range(5):
        for v in range(5):
            if t == v:
                continue
            folder = f"{m}-t{t}-v{v}"
            print(folder)
            with open(ROOT / folder / "metrics.pkl", "rb") as fp:
                pick = pickle.load(fp)
            pick = pick['test']
            
            df = pd.DataFrame()
            
            x = [float(x) for x in pick['count_per_cf'].keys()]
            y = [x['mae'] for x in pick['count_per_cf'].values()]
            mae = np.trapz(y, x, 0.01)
            df['amae'] = [mae]

            y = [x['rmse'] for x in pick['count_per_cf'].values()]
            rmse = np.trapz(y, x, 0.01)
            df['armse'] = [rmse]

            y = [x['rsq'] for x in pick['count_per_cf'].values()]
            rsq = np.trapz(y, x, 0.01)
            df['arsq'] = [rsq]
            df.to_csv(ROOT / folder / "aver_counts.csv", index=False, sep=";")
