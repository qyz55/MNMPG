
# MNMPG

## Note
 This codebase accompanies the paper submission "Credit Assignment with Meta-Policy Gradient for Multi-Agent ReinforcementLearning", and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced.

## Installation instructions

Please follow the instructions on [SMAC](https://github.com/oxwhirl/smac).
## Run an experiment 

To run MNMPG with simple utility network, run:

```shell
export CUDA_VISIBLE_DEVICES="0" && python3 src/main.py --config=qmix_smac_hierarchical --env-config=sc2 with env_args.map_name=6h_vs_8z t_max=5050000
```

To run one layer MNMPG with [RODE](https://github.com/TonghanWang/RODE) utility network, run:

```
export CUDA_VISIBLE_DEVICES="0" && python3 src/main.py --config=hierarchical_rode --env-config=sc2 with env_args.map_name=corridor t_max=5050000 meta_role=False
```

To run two-layer MNMPG with RODE utility network, run:

```
export CUDA_VISIBLE_DEVICES="0" && python3 src/main.py --config=hierarchical_rode --env-config=sc2 with env_args.map_name=corridor t_max=5050000
```


