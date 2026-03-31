#!/bin/bash

# 脚本路径
TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT="test_chunk_gated_delta_rule_single.py"

# ====================== 执行区======================

# 算子调测
run_single() {
    echo "===== 执行用例算子调测 ====="
    python3 -m pytest -rA -s $TEST_CHUNK_GATED_DELTA_RULE_SINGLE_SCRIPT -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [参数]"
    echo "参数说明："
    echo "  single    执行单算子用例调测"
    echo "  help      显示本帮助信息"
    echo "示例："
    echo "  $0 single  # 执行single模式"
}

# ====================== 主逻辑 ======================
# 检查传入的参数数量
if [ $# -ne 1 ]; then
    echo "错误：必须传入且仅传入一个参数（single/help）"
    show_help
    exit 1
fi

# 根据参数执行对应函数
case "$1" in
    single)
        run_single
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知参数 '$1'，仅支持 single/help"
        show_help
        exit 1
        ;;
esac

exit 0
