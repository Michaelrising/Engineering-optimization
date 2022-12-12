nohup python -u main.py --filepath '../Data/Lot1.sch'  --acnet 'mlp' --device 'cuda:0' > ../log/train1_mlp1_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1.sch' --acnet 'cnn' --device 'cuda:0'  > ../log/train1_cnn1_normalized.log 2>&1 &


