python crossval2pcdet.py ./augmented ./out_03-29/rnd_d-t1-v0.csv ./data_05-26/rnd_d-t1-v0 --augs A1minus45 A1minus90 A1plus45 A1plus90 A1step15 OI04 OI-04 &
python crossval2pcdet.py ./augmented ./out_03-29/rnd_d-t1-v2.csv ./data_05-26/rnd_d-t1-v2 --augs A1minus45 A1minus90 A1plus45 A1plus90 A1step15 OI04 OI-04 &
python crossval2pcdet.py ./augmented ./out_03-29/rnd_d-t1-v3.csv ./data_05-26/rnd_d-t1-v3 --augs A1minus45 A1minus90 A1plus45 A1plus90 A1step15 OI04 OI-04 &
python crossval2pcdet.py ./augmented ./out_03-29/rnd_d-t1-v4.csv ./data_05-26/rnd_d-t1-v4 --augs A1minus45 A1minus90 A1plus45 A1plus90 A1step15 OI04 OI-04 &
wait
echo "Done"
