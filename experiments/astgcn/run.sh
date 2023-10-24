# python experiments/astgcn/main.py --device cuda:0 --dataset GLA --years 2019 --model_name astgcn --seed 2018 --bs 16 --patience 50 --wdecay 0

# python experiments/astgcn/main.py --device cuda:0 --dataset GBA --years 2019 --model_name astgcn --seed 2018 --bs 40 --patience 50

# python experiments/astgcn/main.py --device cuda:0 --dataset SD --years 2019 --model_name astgcn --seed 2018 --bs 64 --patience 50

### New dataset

python experiments/astgcn/main.py --device cuda:0 --dataset METR_LA --years all --model_name astgcn --seed 2018 --bs 4
# python experiments/astgcn/main.py --device cuda:0 --dataset PEMS_BAY --years all --model_name astgcn --seed 2018 --bs 4
# python experiments/astgcn/main.py --device cuda:0 --dataset PEMS04 --years all --model_name astgcn --seed 2018 --bs 4
# python experiments/astgcn/main.py --device cuda:0 --dataset PEMS08 --years all --model_name astgcn --seed 2018 --bs 4