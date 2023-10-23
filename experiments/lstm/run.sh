# python experiments/lstm/main.py --device cuda:0 --dataset CA --years 2019 --model_name lstm --seed 2018 --bs 2

# python experiments/lstm/main.py --device cuda:0 --dataset GLA --years 2019 --model_name lstm --seed 2018 --bs 4

# python experiments/lstm/main.py --device cuda:0 --dataset GBA --years 2019 --model_name lstm --seed 2018 --bs 4

# python experiments/lstm/main.py --device cuda:0 --dataset SD --years 2019 --model_name lstm --seed 2018 --bs 4


### New dataset

# python experiments/lstm/main.py --device cuda:0 --dataset METR_LA --years all --model_name lstm --seed 2018 --bs 4
# python experiments/lstm/main.py --device cuda:0 --dataset PEMS_BAY --years all --model_name lstm --seed 2018 --bs 4
python experiments/lstm/main.py --device cuda:0 --dataset PEMS04 --years all --model_name lstm --seed 2018 --bs 4
# python experiments/lstm/main.py --device cuda:0 --dataset PEMS08 --years all --model_name lstm --seed 2018 --bs 4