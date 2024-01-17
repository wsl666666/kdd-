#!/bin/bash

# /home/xingzguo/git_project/DynMixer/scripts/run-exp-3-all.sh
# nohup /home/xingzguo/git_project/DynMixer/scripts/run-exp-3-all.sh > /home/xingzguo/git_project/DynMixer/log/nohup-run-exp-3-all-$(date).log 2>&1 &

cd /home/xingzguo/git_project/DynMixer/src/Mixer
export CUBLAS_WORKSPACE_CONFIG=:4096:8

function run_exp_ppr_bench () {
    local exp_name=$1
    local graph_dataset_name=$2
    local graph_snapshot_basetime=$3
    local graph_snapshot_interval=$4
    local total_sampled_node=$5
    local hidden_size_mlps=$6
    local min_epoch_train=$7
    local clf_alpha=$8
    # local random_seeds=$9
    random_seeds="621 622 623"

    data_strategy=fix-all
    model_max_train_epochs=1000
    num_mlps=2
    
    optim_lr=0.05
    drop_r=0.15

    ppe_out_dim=512
    pprgo_topk=32
    mixrank_mode=null
    mixrank_num_samples=0
    polyrank_mode=null
    polyrank_order=0

    train_per_lb=0.7
    dev_per_lb=0.1
    test_per_lb=0.2

    
    # random_seed="621"
    n_cpu=$(nproc --all)
    
    ### PPR benchmark warmup for numba jit
    for _local_random_seed in 621;
    do
        for alpha in 0.95;
        do
            python main_ppr_benchmark.py \
                --exp_name $exp_name \
                --graph_dataset_name $graph_dataset_name \
                `#--local_proj_data_dir PATH` \
                `#--local_proj_results_dir PATH` \
                --graph_snapshot_basetime $graph_snapshot_basetime \
                --graph_snapshot_interval $graph_snapshot_interval \
                --alpha $alpha \
                --is_dangling_avoid \
                --use_incrmt_ppr \
                --data_strategy $data_strategy \
                --train_per_lb $train_per_lb \
                --dev_per_lb $dev_per_lb \
                --test_per_lb $test_per_lb \
                --total_sampled_node 2 \
                --rs $_local_random_seed \
                --n_cpu $n_cpu 
        done;
    done;

    ### PPR benchmark (use incremental ppr)
    for _local_random_seed in $random_seeds;
    do
        for alpha in $clf_alpha;
        do
            echo  "exp marker: increment seed: $_local_random_seed"
            python main_ppr_benchmark.py \
                --exp_name $exp_name \
                --graph_dataset_name $graph_dataset_name \
                `#--local_proj_data_dir PATH` \
                `#--local_proj_results_dir PATH` \
                --graph_snapshot_basetime $graph_snapshot_basetime \
                --graph_snapshot_interval $graph_snapshot_interval \
                --alpha $alpha \
                --is_dangling_avoid \
                --use_incrmt_ppr \
                --data_strategy $data_strategy\
                --train_per_lb $train_per_lb \
                --dev_per_lb $dev_per_lb \
                --test_per_lb $test_per_lb \
                --total_sampled_node $total_sampled_node \
                --rs $_local_random_seed \
                --n_cpu $n_cpu 
        done;
    done;

    ### PPR benchmark (use non-incremental ppr)
    for _local_random_seed in $random_seeds;
    do
        for alpha in $clf_alpha;
        do
            echo  "exp marker: non-increment seed: $_local_random_seed"
            python main_ppr_benchmark.py \
                --exp_name $exp_name \
                --graph_dataset_name $graph_dataset_name \
                `#--local_proj_data_dir PATH` \
                `#--local_proj_results_dir PATH` \
                --graph_snapshot_basetime $graph_snapshot_basetime \
                --graph_snapshot_interval $graph_snapshot_interval \
                --alpha $alpha \
                --is_dangling_avoid \
                --data_strategy $data_strategy\
                --train_per_lb $train_per_lb \
                --dev_per_lb $dev_per_lb \
                --test_per_lb $test_per_lb \
                --total_sampled_node $total_sampled_node \
                --rs $_local_random_seed \
                --n_cpu $n_cpu 
        done;
    done;

}

exp_version=301


graph_dataset_name=BlogCatalog
exp_name=exp-3-$graph_dataset_name-debug-$exp_version
graph_snapshot_basetime=85871.0
graph_snapshot_interval=21469.0
total_sampled_node=1000 # approx before ceil
hidden_size_mlps="32,16"
min_epoch_train=20
clf_alpha=0.15
random_seeds=" "
run_exp_ppr_bench   $exp_name \
            $graph_dataset_name \
            $graph_snapshot_basetime \
            $graph_snapshot_interval \
            $total_sampled_node \
            $hidden_size_mlps \
            $min_epoch_train \
            $clf_alpha \
            $random_seeds
            


graph_dataset_name=DBLP
exp_name=exp-3-$graph_dataset_name-debug-$exp_version
graph_snapshot_basetime=26433.0
graph_snapshot_interval=6609.0
total_sampled_node=200 # approx before ceil
hidden_size_mlps="32,16"
min_epoch_train=20
clf_alpha=0.15
random_seeds=" "
run_exp_ppr_bench   $exp_name \
            $graph_dataset_name \
            $graph_snapshot_basetime \
            $graph_snapshot_interval \
            $total_sampled_node \
            $hidden_size_mlps \
            $min_epoch_train \
            $clf_alpha \
            $random_seeds


graph_dataset_name=Physics
exp_name=exp-3-$graph_dataset_name-debug-$exp_version
graph_snapshot_basetime=123981.0
graph_snapshot_interval=30996.0
total_sampled_node=200 # approx before ceil
hidden_size_mlps="32,16"
min_epoch_train=20
clf_alpha=0.15
random_seeds=" "
run_exp_ppr_bench   $exp_name \
            $graph_dataset_name \
            $graph_snapshot_basetime \
            $graph_snapshot_interval \
            $total_sampled_node \
            $hidden_size_mlps \
            $min_epoch_train \
            $clf_alpha \
            $random_seeds


# graph_dataset_name=WikiCS
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=108061.0
# graph_snapshot_interval=27016.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=20
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds

# graph_dataset_name=CS
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=108061.0
# graph_snapshot_interval=27016.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=20
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds
            
# graph_dataset_name=cora
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=2638.0
# graph_snapshot_interval=660.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=50
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds


# graph_dataset_name=citeseer
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=2275.0
# graph_snapshot_interval=569.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=50
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds

# graph_dataset_name=pubmed
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=22161.0
# graph_snapshot_interval=5541.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=50
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds


# graph_dataset_name=flickr
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=224938.0
# graph_snapshot_interval=56235.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=100
# clf_alpha=0.3
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds


# graph_dataset_name=ogbn-arxiv
# exp_name=exp-3-$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=578899.0
# graph_snapshot_interval=144725.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="512,256"
# min_epoch_train=100
# clf_alpha=0.15
# random_seeds=" "
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha \
#             $random_seeds


# graph_dataset_name=reddit2
# exp_name=$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=20000000.0 #899 755.0
# graph_snapshot_interval=500000.0
# total_sampled_node=2000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=70
# clf_alpha=0.15
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha


# graph_dataset_name=chameleon
# exp_name=$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=10000.0
# graph_snapshot_interval=6000.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=60
# clf_alpha=0.15
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha

# graph_dataset_name=film
# exp_name=$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=10000.0
# graph_snapshot_interval=5000.0
# total_sampled_node=1000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=50
# clf_alpha=0.15
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha


# graph_dataset_name=squirrel
# exp_name=$graph_dataset_name-debug-$exp_version
# graph_snapshot_basetime=150000.0
# graph_snapshot_interval=15000.0
# total_sampled_node=2000 # approx before ceil
# hidden_size_mlps="32,16"
# min_epoch_train=30
# clf_alpha=0.15
# run_exp_ppr_bench   $exp_name \
#             $graph_dataset_name \
#             $graph_snapshot_basetime \
#             $graph_snapshot_interval \
#             $total_sampled_node \
#             $hidden_size_mlps \
#             $min_epoch_train \
#             $clf_alpha

echo "finish all exit"
exit 0