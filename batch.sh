nohup python -u main.py --filepath '../Data/Lot1.sch'  --acnet 'mlp'  > ../log/train1_mlp_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1003.sch'  --acnet 'mlp'  > ../log/train1003_mlp_normalized.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1004.sch'  --acnet 'mlp'  > ../log/train1004_mlp_normalized.log 2>&1 &


nohup python -u main.py --filepath '../Data/Lot1.sch' --acnet 'cnn'  > ../log/train1_cnn.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1003.sch'  --acnet 'cnn'  > ../log/train1003_cnn.log 2>&1 &
nohup python -u main.py --filepath '../Data/Lot1004.sch'  --acnet 'cnn'  > ../log/train1004_cnn.log 2>&1 &

