nohup python -u main.py --filepath '../Data/Lot1.sch'  --acnet 'mlp' --device 'cuda:0' > ../log/train1_mlp_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1003.sch'  --acnet 'mlp' --device 'cuda:0'  > ../log/train1003_mlp_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1004.sch'  --acnet 'mlp' --device 'cuda:0'  > ../log/train1004_mlp_normalized.log 2>&1 &


nohup python -u main.py --filepath '../Data/Lot1.sch' --acnet 'cnn' --device 'cuda:0'  > ../log/train1_cnn_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1003.sch'  --acnet 'cnn' --device 'cuda:0' > ../log/train1003_cnn_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1004.sch'  --acnet 'cnn' --device 'cuda:0' > ../log/train1004_cnn_normalized.log 2>&1 &

