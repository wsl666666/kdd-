#!/bin/bash

# /home/xingzguo/git_project/DynMixer/scripts/run-exp-1-all.sh
# nohup /home/xingzguo/git_project/DynMixer/scripts/run-exp-1-all.sh > /home/xingzguo/git_project/DynMixer/log/nohup-run-exp-1-all-$(date).log 2>&1 &

cd /home/xingzguo/git_project/DynMixer/src/Mixer
export CUBLAS_WORKSPACE_CONFIG=:4096:8

function run_exp_1 () {
    local exp_name=$1
    local graph_dataset_name=$2
    local graph_snapshot_basetime=$3
    local graph_snapshot_interval=$4
    local total_sampled_node=$5
    local hidden_size_mlps=$6
    local min_epoch_train=$7
    local clf_alpha=$8

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

    # random_seeds="621 622 623"
    random_seeds="625"
    n_cpu=$(nproc --all)
    
    # ### warmup for numba jit
    # for _local_random_seed in 621;
    # do
    #     for alpha in 0.10;
    #     do
    #         python main_ppr_benchmark.py \
    #             --exp_name $exp_name \
    #             --graph_dataset_name $graph_dataset_name \
    #             `#--local_proj_data_dir PATH` \
    #             `#--local_proj_results_dir PATH` \
    #             --graph_snapshot_basetime $graph_snapshot_basetime \
    #             --graph_snapshot_interval $graph_snapshot_interval \
    #             --alpha $alpha \
    #             --is_dangling_avoid \
    #             `#--use_incrmt_ppr PATH` \
    #             --data_strategy $data_strategy\
    #             --train_per_lb $train_per_lb \
    #             --dev_per_lb $dev_per_lb \
    #             --test_per_lb $test_per_lb \
    #             --total_sampled_node 2 \
    #             --rs $_local_random_seed \
    #             --n_cpu $n_cpu 
    #     done;
    # done;

    # GoPPE 
    for random_seed in $random_seeds;
    do
        for aggregate_type in  goppe ;
        do
            for ppr_algo in fista; # ista
            do
                for alpha in $clf_alpha;
                do
                    python main.py \
                        --exp_name $exp_name \
                        --graph_dataset_name $graph_dataset_name \
                        `#--local_proj_data_dir PATH` \
                        `#--local_proj_results_dir PATH` \
                        --graph_snapshot_basetime $graph_snapshot_basetime \
                        --graph_snapshot_interval $graph_snapshot_interval \
                        --aggregate_type $aggregate_type \
                        --alpha $alpha \
                        --ppr_algo $ppr_algo \
                        --is_dangling_avoid \
                        `#--use_incrmt_ppr PATH` \
                        --num_mlps $num_mlps \
                        --hidden_size_mlps $hidden_size_mlps \
                        --drop_r $drop_r \
                        --ppe_out_dim $ppe_out_dim \
                        --pprgo_topk $pprgo_topk \
                        --mixrank_mode $mixrank_mode \
                        --mixrank_num_samples $mixrank_num_samples \
                        --polyrank_mode $polyrank_mode \
                        --polyrank_order $polyrank_order \
                        --optim_lr $optim_lr \
                        --model_max_train_epochs $model_max_train_epochs \
                        --min_epoch_train  $min_epoch_train \
                        --data_strategy $data_strategy\
                        --train_per_lb $train_per_lb \
                        --dev_per_lb $dev_per_lb \
                        --test_per_lb $test_per_lb \
                        --total_sampled_node $total_sampled_node \
                        --is_retrain_each_snapshot \
                        --rs $random_seed \
                        --use_cuda \
                        --use_torch_sparse \
                        --use_verbose \
                        `#--use_graph_ppv_debug` \
                        --n_cpu $n_cpu 
                done;
            done;
        done;
    done;

}

exp_version=1001 # for debug only

graph_dataset_name=cora
exp_name=exp-1-$graph_dataset_name-debug-$exp_version
graph_snapshot_basetime=2638.0
graph_snapshot_interval=660.0
total_sampled_node=1000 # approx before ceil
hidden_size_mlps="32,16"
min_epoch_train=100
clf_alpha=0.15
run_exp_1   $exp_name \
            $graph_dataset_name \
            $graph_snapshot_basetime \
            $graph_snapshot_interval \
            $total_sampled_node \
            $hidden_size_mlps \
            $min_epoch_train \
            $clf_alpha


echo "finish all exit"
exit 0