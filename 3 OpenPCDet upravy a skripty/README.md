**pcdet/**  
Obsahuje úpravy frameworku OpenPCDet. Po instalaci frameworku nahraďte danými soubory. Obsahuje i úpravy pro podporu Point-RCNN

**tools/**  
Vlastní training skripty. Spouštějte `run_train.py` a `run_test.py`, pro trénink vnořené CV je případně možné použít bash skript:  
`./sh-train-nested.sh ./cfgs/base_second_mdl.yaml $výstup 300 $metoda 100 25 ap_sum_0.3 max`

**tools/avcounts.py**  
Výpočet AUC verzí početních metrik z metrics.pkl