#!/bin/bash
# 分阶段实验执行脚本
# 使用方法: bash scripts/run_experiments.sh [phase] [experiment_name]

PHASE=${1:-"all"}  # phase1, phase2, phase3, phase4, all
EXP_NAME=${2:-"default"}

BASE_DIR="./output"
LOG_DIR="./logs"

# 创建必要的目录
mkdir -p ${BASE_DIR}
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "开始执行实验: Phase ${PHASE}, Experiment: ${EXP_NAME}"
echo "=========================================="

# ============================================
# Phase 1: CI2P模块实验
# ============================================
if [ "$PHASE" == "phase1" ] || [ "$PHASE" == "all" ]; then
    echo ""
    echo ">>> Phase 1: CI2P模块实验"
    echo "----------------------------------------"
    
    # 1.1 基线模型训练
    echo "训练基线模型..."
    python tracking/train.py \
        --script mmtrack \
        --config baseline \
        --save_dir ${BASE_DIR}/baseline_${EXP_NAME} \
        --mode multiple \
        --nproc_per_node 2 \
        --use_wandb 1 \
        2>&1 | tee ${LOG_DIR}/baseline_train_${EXP_NAME}.log
    
    # 1.2 CI2P模型训练
    echo "训练CI2P模型..."
    python tracking/train.py \
        --script mmtrack \
        --config ci2p_ablation \
        --save_dir ${BASE_DIR}/ci2p_${EXP_NAME} \
        --mode multiple \
        --nproc_per_node 2 \
        --use_wandb 1 \
        2>&1 | tee ${LOG_DIR}/ci2p_train_${EXP_NAME}.log
    
    # 1.3 测试基线模型
    echo "测试基线模型..."
    python tracking/test.py \
        --tracker_name mmtrack \
        --tracker_param baseline \
        --dataset_name lasot_lang \
        --threads 8 \
        --num_gpus 2 \
        2>&1 | tee ${LOG_DIR}/baseline_test_${EXP_NAME}.log
    
    # 1.4 测试CI2P模型
    echo "测试CI2P模型..."
    python tracking/test.py \
        --tracker_name mmtrack \
        --tracker_param ci2p_ablation \
        --dataset_name lasot_lang \
        --threads 8 \
        --num_gpus 2 \
        2>&1 | tee ${LOG_DIR}/ci2p_test_${EXP_NAME}.log
    
    # 1.5 分析结果
    echo "分析Phase 1结果..."
    python tracking/analysis_results.py \
        --baseline_dir ${BASE_DIR}/baseline_${EXP_NAME} \
        --ci2p_dir ${BASE_DIR}/ci2p_${EXP_NAME} \
        --output ${LOG_DIR}/phase1_analysis_${EXP_NAME}.txt
fi

# ============================================
# Phase 2: Transformer-XL实验
# ============================================
if [ "$PHASE" == "phase2" ] || [ "$PHASE" == "all" ]; then
    echo ""
    echo ">>> Phase 2: Transformer-XL解码器实验"
    echo "----------------------------------------"
    
    # 2.1 Transformer-XL模型训练
    echo "训练Transformer-XL模型..."
    python tracking/train.py \
        --script mmtrack \
        --config transformer_xl \
        --save_dir ${BASE_DIR}/transformer_xl_${EXP_NAME} \
        --mode multiple \
        --nproc_per_node 2 \
        --use_wandb 1 \
        2>&1 | tee ${LOG_DIR}/transformer_xl_train_${EXP_NAME}.log
    
    # 2.2 测试Transformer-XL模型
    echo "测试Transformer-XL模型..."
    python tracking/test.py \
        --tracker_name mmtrack \
        --tracker_param transformer_xl \
        --dataset_name lasot_lang \
        --threads 8 \
        --num_gpus 2 \
        2>&1 | tee ${LOG_DIR}/transformer_xl_test_${EXP_NAME}.log
    
    # 2.3 长序列测试
    echo "测试长序列性能..."
    python tracking/test.py \
        --tracker_name mmtrack \
        --tracker_param transformer_xl \
        --dataset_name lasot_lang \
        --long_sequence_test \
        --threads 8 \
        --num_gpus 2 \
        2>&1 | tee ${LOG_DIR}/transformer_xl_longseq_${EXP_NAME}.log
fi

# ============================================
# Phase 3: ETT Tokenizer实验
# ============================================
if [ "$PHASE" == "phase3" ] || [ "$PHASE" == "all" ]; then
    echo ""
    echo ">>> Phase 3: ETT Tokenizer实验"
    echo "----------------------------------------"
    
    # 3.1 ETT模型训练
    echo "训练ETT Tokenizer模型..."
    python tracking/train.py \
        --script mmtrack \
        --config ett_tokenizer \
        --save_dir ${BASE_DIR}/ett_${EXP_NAME} \
        --mode multiple \
        --nproc_per_node 2 \
        --use_wandb 1 \
        2>&1 | tee ${LOG_DIR}/ett_train_${EXP_NAME}.log
    
    # 3.2 测试ETT模型
    echo "测试ETT Tokenizer模型..."
    python tracking/test.py \
        --tracker_name mmtrack \
        --tracker_param ett_tokenizer \
        --dataset_name lasot_lang \
        --threads 8 \
        --num_gpus 2 \
        2>&1 | tee ${LOG_DIR}/ett_test_${EXP_NAME}.log
fi

# ============================================
# Phase 4: 完整集成实验
# ============================================
if [ "$PHASE" == "phase4" ] || [ "$PHASE" == "all" ]; then
    echo ""
    echo ">>> Phase 4: 完整集成实验"
    echo "----------------------------------------"
    
    # 4.1 完整模型训练
    echo "训练完整模型（所有模块）..."
    python tracking/train.py \
        --script mmtrack \
        --config full_model \
        --save_dir ${BASE_DIR}/full_model_${EXP_NAME} \
        --mode multiple \
        --nproc_per_node 2 \
        --use_wandb 1 \
        2>&1 | tee ${LOG_DIR}/full_model_train_${EXP_NAME}.log
    
    # 4.2 完整消融实验
    echo "运行完整消融实验..."
    bash scripts/run_ablation_studies.sh ${EXP_NAME}
    
    # 4.3 SOTA对比实验
    echo "运行SOTA对比实验..."
    bash scripts/run_sota_comparison.sh ${EXP_NAME}
fi

echo ""
echo "=========================================="
echo "实验完成: Phase ${PHASE}, Experiment: ${EXP_NAME}"
echo "结果保存在: ${BASE_DIR}"
echo "日志保存在: ${LOG_DIR}"
echo "=========================================="

