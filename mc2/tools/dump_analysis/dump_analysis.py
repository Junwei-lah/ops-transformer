#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
# 

# !
#  \file dump_analysis.py
#  \brief

import os
import sys
import csv
import logging
from collections import Counter
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
    )


@dataclass
class WinData:
    win_data_list_01: np.ndarray | list
    win_data_list_02: np.ndarray | list
    win_data_01: int
    win_data_02: int
    card_num_class: int
    bs_class: int
    k_class: int


# 识别使用了多少个核,通过查看没512B的前9*4B的位置是否全为0来判断
def analysis_core_num(arr_func: np.ndarray) -> int:
    per_core = 128 # 512B转为int32
    judge_arr = [0, 0, 0, 0, 0, 0, 0, 0, 0] # 标识区内全为0,识别是否走到最后一个核
    card_num_func = 0
    for i in range(100):
        if (judge_arr == arr_func[i * per_core:(len(judge_arr) + i * per_core)]).all():
            card_num_func = i
            break
    return card_num_func


# 识别是否跑了单算子,识别dis&com使用核数是否相同，将dis的使用核数与com的使用核数对比
def compare_core_num(dis_core_num_func: int, com_core_num_func: int, card_num_func: int) -> dict:
    core_error_dict = {}
    logging.info("1. 该卡dispatch&combine使用核数为 dispatch:%d, combine:%d", dis_core_num_func, com_core_num_func)
    if (dis_core_num_func == 0 or com_core_num_func == 0):
        logging.info("1. 仅跑了单个算子")
        return core_error_dict
    else:
        if (dis_core_num_func != com_core_num_func):
            core_error_dict[f"d{card_num_func}"] = \
                f"dispatch的使用核数:{dis_core_num_func} 与 combine的使用核数:{com_core_num_func} 应该相等"
            logging.warning("dispatch的使用核数:%d 与 combine的使用核数:(%d) 应该相等",
                        dis_core_num_func, com_core_num_func)
            return core_error_dict
        else:
            return core_error_dict


# 获取每个核的执行次数，将出现次数最多的执行次数作为该卡的执行次数，并将与该卡的执行次数不同的核记录下来
def analysis_run_num(arr_func: np.ndarray, card_num_func: int):
    per_core = 128
    run_num_list = []
    diff_indices = [] # 存放与大多数执行次数不同的核的下标
    for i in range(card_num_func):
        run_num_list.append(arr_func[6 + (i * per_core)])
    run_num_count = Counter(run_num_list)
    max_run_num = max(run_num_count, key=run_num_count.get)
    diff_indices = [idx for idx, val in enumerate(run_num_list) if val != max_run_num]
    return run_num_list, diff_indices, max_run_num


# 对比com&dis的执行次数, 并将与次数最多的执行次数不同的核打印，并根据dsi&com的执行次数关系给出结论
def compare_run_num(parms: WinData, dis_diff_indices_func: list, com_diff_indices_func: list) -> dict:
    dis_run_num_list_func = parms.win_data_list_01
    com_run_num_list_func = parms.win_data_list_02
    max_dis_run_num_func = parms.win_data_01
    max_com_run_num_func = parms.win_data_02
    card_num_func = parms.card_num_class
    run_num_error_dict = {}
    dis_max_run_num = len(dis_run_num_list_func) - len(dis_diff_indices_func)
    com_max_run_num = len(com_run_num_list_func) - len(com_diff_indices_func)
    if len(dis_diff_indices_func) != 0:
        for x in dis_diff_indices_func:
            run_num_error_dict[f"d{card_num_func}_{x}core_dispatch"] = (
                                f"{x} core run_num should be {max_dis_run_num_func}"
                                f"({dis_max_run_num}/{len(dis_run_num_list_func)}) but got {dis_run_num_list_func[x]}"
                                )
    if len(com_diff_indices_func) != 0:
        for x in com_diff_indices_func:
            run_num_error_dict[f"d{card_num_func}_{x}core_combine"] = (
                                f"{x} core run_num should be {max_com_run_num_func}"
                                f"({com_max_run_num}/{len(com_run_num_list_func)}) but got {com_run_num_list_func[x]}"
                                )

    if max_dis_run_num_func == max_com_run_num_func:
        logging.warning("dispatch执行次数:%d = combine执行次数:%d ,挂在combine上",
                        max_dis_run_num_func, max_com_run_num_func)
        run_num_error_dict[f"d{card_num_func}"] = (
                                                f"dispatch执行次数:{max_dis_run_num_func} = "
                                                f"combine执行次数:{max_dis_run_num_func},挂在combine上"
                                                )
    elif max_dis_run_num_func == (max_com_run_num_func + 1):
        logging.warning("dispatch执行次数:%d = combine执行次数:%d + 1,挂在dispatch上",
                        max_dis_run_num_func, max_com_run_num_func)
        run_num_error_dict[f"d{card_num_func}"] = (
                                                f"dispatch执行次数:{max_dis_run_num_func} = "
                                                f"combine执行次数:{max_dis_run_num_func} + 1,挂在diapatch上"
                                                )
    else:
        logging.warning("dispatch执行次数:%d, combine执行次数:%d 无法判断", max_dis_run_num_func, max_com_run_num_func)
        run_num_error_dict[f"d{card_num_func}"] = (
                f"dispatch执行次数:{max_dis_run_num_func},combine执行次数:{max_dis_run_num_func},无法判断")
    return run_num_error_dict


# 获取每个核的epworldsize,moe专家数,将出现最多次数的epworldsize,moe专家数作为该卡的epworldsize,moe专家数
def get_ep_moe(arr_func: np.ndarray, card_num_func: int):
    per_core = 128
    ep_num = []
    moe_num = []
    for i in range(card_num_func):
        ep_num.append(arr_func[4 + (i * per_core)])
        moe_num.append(arr_func[3 + (i * per_core)])
    ep_num_count = Counter(ep_num)
    moe_num_count = Counter(moe_num)
    max_ep_num = max(ep_num_count, key=ep_num_count.get)
    max_moe_num = max(moe_num_count, key=moe_num_count.get)
    return max_ep_num, max_moe_num


# 获取每个核的执行位置,并以第一个核的0/1标识位作为该卡的0/1标识区,并记录没等到状态的核位置
def get_status_info(arr_func: np.ndarray, card_num_func: int):
    per_core = 128
    list_status = []
    list_0_1_list = []
    unwait_index = [] # 存放没等到状态的核的下标
    for i in range(card_num_func):
        list_status.append(arr_func[1 + (i * per_core)])
        list_0_1_list.append(arr_func[0 + (i * per_core)])
    list_0_1_count = Counter(list_0_1_list)
    max_0_1 = max(list_0_1_count, key=list_0_1_count.get)
    diff_0_1 = [idx for idx, val in enumerate(list_0_1_list) if val != max_0_1]
    unwait_index = [index for index, value in enumerate(list_status) if value == 1]
    return list_status, max_0_1, unwait_index, diff_0_1, list_0_1_list


#判断是否有核0/1标识位与其他核不相等
def get_diff_0_1(arr_func: np.ndarray, diff_0_1: list, card_num_func: int, d_c: str, core_0_1) -> dict:
    diff_error_dict = {}
    if diff_0_1 != []:
        if d_c == "dispatch":
            logging.warning("2.1 dispatch有如下下标的核的状态位与其他核不相等%s,共%d个核",
                        diff_0_1, len(diff_0_1))
        else:
            logging.warning("3.1 combine有如下下标的核的状态位与其他核不相等%s,共%d个核",
                        diff_0_1, len(diff_0_1))
        for x in diff_0_1:
            diff_error_dict[f"d{card_num_func}_{x}core_{str}"] = (
                                f"{x} core 0/1标识位:{arr_func[x]} 与其他核:{core_0_1}不相等"
                                )
    return diff_error_dict


# 判断是否为共享专家卡,来分析该卡dis的总状态位数量
def get_dis_status_num(dis_epworldsize_func: int, dis_moe_num_func: int, all_card_num_func: int, card_num_func: int,
                       share_expert_card_count_func: int):
    status_num = 0
    if ((card_num_func + 1) <= share_expert_card_count_func):
        logging.info("2.1 该卡为共享专家卡")
        status_num = dis_epworldsize_func # 共享专家卡:总状态位数量 = ep_worldsize
    else:
        logging.info("2.1 该卡不为共享专家卡")
        # 非共享专家卡:总状态位数量 = ep_worldsize * 单卡moe专家数
        status_num = int((dis_moe_num_func / (all_card_num_func - share_expert_card_count_func)) * dis_epworldsize_func)
    logging.info("2.1 dispatch总状态位数量:%d", status_num)

    return status_num


# 根据共享专家数量,分析该卡com的总状态位数量
def get_com_status_num(bs_func: int, k_func: int, share_expert_num_func: int, len_status: int):
    status_num = 0
    status_num = bs_func * (k_func + share_expert_num_func) # 总状态位数量=BS*(K+共享专家数)
    logging.info("3.1 combine总状态位数量:%d", status_num)
    if len_status < (status_num * 8):
        logging.error("3.1 计算得出的总状态位*8(%d) 大于combine状态区数据(int32)shape(%d), 请检查BS,K是否输入正确",
                     (status_num * 8), len_status)
    return status_num


# dis状态位分析，通过传入的未等到状态的核下标，在对应的0/1状态区查找具体哪个状态没有等到
def dis_status_analysis(parms: WinData, dis_core_num_func: int, dis_unwait_index_func: list):
    int32_status_data = parms.win_data_list_01
    dis_status_list_func = parms.win_data_list_02
    dis_status_num_func = parms.win_data_01
    dis_0_1_func = parms.win_data_02
    card_num_func = parms.card_num_class
    #状态位数量平均分配给每个核，若不能整除，余数平均分配给前per_core_status_num_any个核
    per_core_status_num = int(dis_status_num_func // dis_core_num_func)
    per_core_status_num_any = int(dis_status_num_func % dis_core_num_func)
    dis_status_core = []
    dis_status_error_dict = {}
    if per_core_status_num_any != 0:
        dis_status_core += [per_core_status_num + 1] * per_core_status_num_any
        dis_status_core += [per_core_status_num] * (dis_core_num_func - per_core_status_num_any)
    else:
        dis_status_core += [per_core_status_num] * dis_core_num_func
    logging.info("2.2 dispatch 中各核分配到状态位数量%s", dis_status_core)
    if dis_unwait_index_func == []:
        return dis_status_error_dict
    logging.warning("2.2 dispatch有如下下标的核没有等到状态%s,共%d个核",
                    dis_unwait_index_func, len(dis_unwait_index_func))
    # 在未等到状态的核的对应0/1状态区查找具体哪个状态没有等到
    for i in dis_unwait_index_func:
        if (sum(dis_status_core[:i + 1]) > dis_status_num_func):
            break
        if (dis_status_list_func[i] != 1):
            continue
        for a in range(dis_status_core[i]):
            if int32_status_data[(sum(dis_status_core[:i + 1]) - dis_status_core[i] + a) * 8] == 0:
                dis_status_error_dict[f"d{card_num_func}_第{i}个核_第{a}状态位_dispatch{dis_0_1_func}"] = (
                                                    f"状态位没有等到")
    return dis_status_error_dict


# com状态位分析，通过传入的未等到状态的核下标，在对应的0/1状态区查找具体哪个状态没有等到
def com_status_analysis(parms: WinData, com_core_num_func: int, share_expert_num_func: int,
                        com_unwait_index_func: list):
    int32_status_data = parms.win_data_list_01
    com_status_list_func = parms.win_data_list_02
    com_status_num_func = parms.win_data_01
    com_0_1_func = parms.win_data_02
    bs_func = parms.bs_class
    k_func = parms.k_class
    card_num_func = parms.card_num_class
    # 状态位数量按BS平均分配给每个核，若不能整除，余数的BS平均分配给前per_core_status_num_any个核
    per_bs_status = k_func + share_expert_num_func # 单个BS的状态位数量
    per_core_status_num = int(bs_func // com_core_num_func)
    per_core_status_num_any = int(bs_func % com_core_num_func)
    com_status_core = []
    com_statu_error_dict = {}
    if per_core_status_num_any != 0:
        com_status_core += [(per_core_status_num + 1) * per_bs_status] * per_core_status_num_any
        com_status_core += [per_core_status_num * per_bs_status] * (com_core_num_func - per_core_status_num_any)
    else:
        com_status_core += [per_core_status_num * per_core_status_num] * com_core_num_func
    logging.info("2.2 combine 中各核分配到状态位数量%s", com_status_core)
    if com_unwait_index_func == []:
        return com_statu_error_dict
    logging.warning("3.2 combine有如下下标的核没有等到状态%s,共%d个核",
                    com_unwait_index_func, len(com_unwait_index_func))
    # 在未等到状态的核的对应0/1状态区查找具体哪个状态没有等到
    for i in com_unwait_index_func:
        if (sum(com_status_core[:i + 1]) > com_status_num_func):
            break
        if (com_status_list_func[i] != 1):
            continue
        for a in range(com_status_core[i]):
            if int32_status_data[(sum(com_status_core[:i + 1]) - com_status_core[i] + a) * 8] == 0:
                com_statu_error_dict[f"d{card_num_func}_第{i}个核_第{a}个状态位_combine{com_0_1_func}区"] = (
                                                    f"状态位没有等到")
    return com_statu_error_dict


# main
bs = int(sys.argv[1])
k = int(sys.argv[2])
share_expert_card_count = int(sys.argv[3])
share_expert_num = int(sys.argv[4])
all_card_num = int(sys.argv[5])
card_num = int(sys.argv[6])
floder_path = sys.argv[7]
soc_version = sys.argv[8]
error_dict = {}
max_dis_run_num = 0
max_com_run_num = 0

if (soc_version == "910_95"):
    perfix = "mc2_"
    endfix = ""
elif (soc_version == "910_93"):
    perfix = "exception_info"
    endfix = ".workspace.1.bin.npy"
else:
    logging.error("soc_version:%d 非法输入, soc_version should be: 910_93 or 910_95", soc_version)

for filename in tqdm(os.listdir(os.path.join(floder_path))):
    if filename.startswith(perfix) and filename.endswith(endfix):
        logging.info("开始分析卡%d数据", card_num)
        file_path = os.path.join(floder_path, filename)
        logging.info("解析文件:%s\n", file_path)
        with open(file_path, "rb") as f:
            arr = np.frombuffer(f.read(), dtype=np.int8)
        last_1M = arr[-1024 * 1024:]
        dis_0_status = last_1M[0:65 * 1024]
        com_0_status = last_1M[64 * 1024:384 * 1024]
        dis_1_status = last_1M[384 * 1024:448 * 1024]
        com_1_status = last_1M[448 * 1024:768 * 1024]
        dis_win_data = last_1M[768 * 1024:818 * 1024]
        com_win_data = last_1M[818 * 1024:868 * 1024]

        int32_arr = last_1M.view(np.int32)
        int32_dis_0_status = dis_0_status.view(np.int32)
        int32_com_0_status = com_0_status.view(np.int32)
        int32_dis_1_status = dis_1_status.view(np.int32)
        int32_com_1_status = com_1_status.view(np.int32)
        int32_dis_win_data = dis_win_data.view(np.int32)
        int32_com_win_data = com_win_data.view(np.int32)

        # 判断卡上 dis&com使用核数是否一致
        dis_core_num = analysis_core_num(int32_dis_win_data)
        com_core_num = analysis_core_num(int32_com_win_data)
        error_dict.update(compare_core_num(dis_core_num, com_core_num, card_num))

        # 执行序分析
        logging.info("1. 开始执行序分析")
        if (dis_core_num != 0):
            dis_run_num_list, dis_diff_indices, max_dis_run_num = analysis_run_num(int32_dis_win_data, dis_core_num)
            logging.info("1. dispatch各核执行次数:%s", dis_run_num_list)
        if (com_core_num != 0):
            com_run_num_list, com_diff_indices, max_com_run_num = analysis_run_num(int32_com_win_data, com_core_num)
            logging.info("1. combine各核执行次数:%s", com_run_num_list)
        if (dis_core_num != 0 and com_core_num != 0):
            run_num_class = WinData(win_data_list_01=dis_run_num_list, win_data_list_02=com_run_num_list,
                                       win_data_01=max_dis_run_num, win_data_02=max_com_run_num,
                                       card_num_class=card_num, bs_class=bs, k_class=k)
            error_dict.update(compare_run_num(run_num_class, dis_diff_indices, com_diff_indices))
            logging.info("1. 执行序分析完成\n")
        else:
            logging.info("1. 单个算子的调用场景不进行执行序分析\n")
        
        # 状态位分析
        if (dis_core_num != 0):
            logging.info("2. 开始dispatch状态位分析")
            dis_epworldsize, dis_moe_num = get_ep_moe(int32_dis_win_data, dis_core_num)
            logging.info("2.1 dispatch_epworldsize:%d, dispatch moe专家数:%d", dis_epworldsize, dis_moe_num)
            dis_status_list, dis_0_1, dis_unwait_index, dis_diff_0_1_list, dis_0_1_list = get_status_info(
                                                                                    int32_dis_win_data, dis_core_num)
            error_dict.update(get_diff_0_1(dis_0_1_list, dis_diff_0_1_list, card_num, "dispatch", dis_0_1))
            logging.info("2.1 dispatch各核执行位置情况:%s", dis_status_list)
            dis_status_num = get_dis_status_num(dis_epworldsize, dis_moe_num, all_card_num, card_num,
                                                share_expert_card_count)
            if (len(int32_dis_0_status) < (dis_status_num * 8)):
                logging.error("2.1 计算得出的总状态位*8(%d) 大于dispatch状态区数据(int32)shape(%d)",
                             (dis_status_num * 8), len(int32_dis_0_status))
            if dis_0_1 == 0:
                logging.info("2.2 dispatch 0区状态区数据:%s", int32_dis_0_status.dtype)
                logging.info("2.2 dispatch 0区状态区数据 shape:%d", len(int32_dis_0_status))
                logging.info("2.2 dispatch 0区状态区数据:%s", int32_dis_0_status)
                dis_status_class = WinData(win_data_list_01=int32_dis_0_status, win_data_list_02=dis_status_list, 
                                              win_data_01=dis_status_num, win_data_02=dis_0_1, card_num_class=card_num,
                                              bs_class=bs, k_class=k)
                error_dict.update(dis_status_analysis(dis_status_class, dis_core_num, dis_unwait_index))
                logging.info("2.3 dispatch状态区分析完成\n")
            elif dis_0_1 == 1:
                logging.info("2.2 dispatch 1区状态区数据:%s", int32_dis_1_status.dtype)
                logging.info("2.2 dispatch 1区状态区数据 shape:%d", len(int32_dis_1_status))
                logging.info("2.2 dispatch 1区状态区数据:%s", int32_dis_1_status)
                dis_status_class = WinData(win_data_list_01=int32_dis_1_status, win_data_list_02=dis_status_list, 
                                              win_data_01=dis_status_num, win_data_02=dis_0_1, card_num_class=card_num,
                                              bs_class=bs, k_class=k)
                error_dict.update(dis_status_analysis(dis_status_class, dis_core_num, dis_unwait_index))
                logging.info("2.3 dispatch状态区分析完成\n")
            else:
                logging.error("2. dispatch 0/1标识位 should be 0/1 but got %d\n", dis_0_1)
        else:
            logging.info("2. 未调用dispatch,不进行dispatch状态区分析\n")
        
        if (com_core_num != 0):
            logging.info("3. 开始combine状态位分析")
            com_epworldsize, com_moe_num = get_ep_moe(int32_com_win_data, com_core_num)
            logging.info("3.1 combine_epworldsize:%d, combine moe专家数:%d", com_epworldsize, com_moe_num)
            com_status_list, com_0_1, com_unwait_index, com_diff_0_1_list, com_0_1_list = get_status_info(
                                                                                    int32_com_win_data, com_core_num)
            error_dict.update(get_diff_0_1(com_0_1_list, com_diff_0_1_list, card_num, "combine", com_0_1))
            logging.info("3.1 combine各核执行位置情况:%s", com_status_list)
            com_status_num = get_com_status_num(bs, k, share_expert_num, len(int32_dis_0_status))
            if com_0_1 == 0:
                logging.info("3.2 combine 0区状态区数据:%s", int32_com_0_status.dtype)
                logging.info("3.2 combine 0区状态区数据 shape:%d", len(int32_com_0_status))
                logging.info("3.2 combine 0区状态区数据:%s", int32_com_0_status)
                com_status_class = WinData(win_data_list_01=int32_com_0_status, win_data_list_02=com_status_list, 
                                              win_data_01=com_status_num, win_data_02=com_0_1, bs_class=bs,
                                              k_class=k, card_num_class=card_num)
                error_dict.update(com_status_analysis(com_status_class, com_core_num, share_expert_num,
                                                      com_unwait_index))
                logging.info("3.3 combine状态区分析完成\n")
            elif com_0_1 == 1:
                logging.info("3.2 combine 1区状态区数据:%s", int32_com_1_status.dtype)
                logging.info("3.2 combine 1区状态区数据 shape:%d", len(int32_com_1_status))
                logging.info("3.2 combine 1区状态区数据:%s", int32_com_1_status)
                com_status_class = WinData(win_data_list_01=int32_com_1_status, win_data_list_02=com_status_list, 
                                              win_data_01=com_status_num, win_data_02=com_0_1, bs_class=bs,
                                              k_class=k, card_num_class=card_num)
                error_dict.update(com_status_analysis(com_status_class, com_core_num, share_expert_num,
                                                      com_unwait_index))
                logging.info("3.3 combine状态区分析完成\n")
            else:
                logging.error("3. combine 0/1标识位 should be 0/1 but got %d\n", com_0_1)
        else:
            logging.info("3. 未调用com,不进行com状态区分析\n")

        # 输出
        logging.info("4. 数据归档")
        if (dis_core_num != 0):
            file_status_list_eixsts = os.path.exists("win_status_list.csv")
            file_eixsts = os.path.exists("win_data.csv")
            file_list_eixsts = os.path.exists("win_data_list.csv")
            dis_core_info = pd.DataFrame([[dis_core_num, dis_epworldsize, dis_moe_num, dis_0_1]],
                                        columns=['使用核数', 'ep_worldsize', 'moe专家数', '0/1标识'], 
                                        index=[f"d{card_num}_dispatch"])
            dis_run_num_list_info = pd.DataFrame([dis_run_num_list], index=[f"d{card_num}_dispatch各核执行次数数据"])
            dis_run_num_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=not file_list_eixsts,
                                        encoding="gbk")
            dis_core_info.to_csv("win_data.csv", index=True, mode='a', header=not file_eixsts, encoding="gbk")
            dis_status_list_info = pd.DataFrame([dis_status_list], index=[f"d{card_num}_dispatch各核执行位置数据"])
            dis_status_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=False, encoding="gbk")
            dis_0_1_list_info = pd.DataFrame([dis_0_1_list], index=[f"d{card_num}_dispatch各核0/1标识区数据"])
            dis_0_1_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=False, encoding="gbk")
            if dis_0_1 == 0:
                int32_dis_0_status_info = pd.DataFrame([int32_dis_0_status],
                                                        index=[f"d{card_num}_dispatch 0区状态区数据"])
                int32_dis_0_status_info.to_csv("win_status_list.csv", index=True, mode='a',
                                                header=not file_status_list_eixsts, encoding="gbk")
            else:
                int32_dis_1_status_info = pd.DataFrame([int32_dis_1_status],
                                                        index=[f"d{card_num}_dispatch 1区状态区数据"])
                int32_dis_1_status_info.to_csv("win_status_list.csv", index=True, mode='a',
                                                header=not file_status_list_eixsts, encoding="gbk")

        if (com_core_num != 0):
            file_status_list_eixsts = os.path.exists("win_status_list.csv")
            file_eixsts = os.path.exists("win_data.csv")
            file_list_eixsts = os.path.exists("win_data_list.csv")
            com_core_info = pd.DataFrame([[com_core_num, com_epworldsize, com_moe_num, com_0_1]],
                                        columns=['使用核数', 'ep_worldsize', 'moe专家数', '0/1标识'], 
                                        index=[f"d{card_num}_combine"])
            com_core_info.to_csv("win_data.csv", index=True, mode='a', header=not file_eixsts, encoding="gbk")
            com_run_num_list_info = pd.DataFrame([com_run_num_list], index=[f"d{card_num}_combine各核执行次数数据"])
            com_run_num_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=not file_list_eixsts,
                                        encoding="gbk")
            com_status_list_info = pd.DataFrame([com_status_list], index=[f"d{card_num}_combine各核执行位置数据"])
            com_status_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=False, encoding="gbk")
            com_0_1_list_info = pd.DataFrame([com_0_1_list], index=[f"d{card_num}_combine各核0/1标识区数据"])
            com_0_1_list_info.to_csv("win_data_list.csv", index=True, mode='a', header=False, encoding="gbk")
            if com_0_1 == 0:
                int32_com_0_status_info = pd.DataFrame([int32_com_0_status],
                                                        index=[f"d{card_num}_combine 0区状态区数据"])
                int32_com_0_status_info.to_csv("win_status_list.csv", index=True, mode='a',
                                                header=not file_status_list_eixsts, encoding="gbk")
            else:
                int32_com_1_status_info = pd.DataFrame([int32_com_1_status],
                                                        index=[f"d{card_num}_combine 1区状态区数据"])
                int32_com_1_status_info.to_csv("win_status_list.csv", index=True, mode='a',
                                                header=not file_status_list_eixsts, encoding="gbk")

        logging.info("4. 该卡的dispatch&combine的使用核数、ep_worldsize、moe专家数、0/1标识区数据已归档至win_data.csv")
        logging.info("4. 该卡的中各核的dispatch&combine的执行次数、执行位置、0/1标识区数据已归档至win_data_list.csv")
        logging.info("4. 该卡的中所使用的dispatch&combine状态区数据已归档至win_status_list.csv")
        file_error_eixsts = os.path.exists("win_analysis_error.csv")
        error_rows = [[key, value] for key, value in error_dict.items()]
        df = pd.DataFrame(error_rows, columns=["card_num", "error"])
        df.to_csv("win_analysis_error.csv", index=True, mode='a', header=not file_error_eixsts, encoding="gbk")
        logging.info("4. 分析出的错误详细信息归档至win_analysis_error.csv\n")
        file_all_card_eixsts = os.path.exists("win_all_card_run_num.csv")
        all_card_run_num_info = pd.DataFrame([[max_dis_run_num, max_com_run_num]],
                            columns=['dispatch执行次数', 'combine执行次数'], 
                            index=[f"d{card_num}"])
        all_card_run_num_info.to_csv("win_all_card_run_num.csv", index=True, mode='a',
                                    header=not file_all_card_eixsts, encoding="gbk")
        # 当分析到最后一张卡时，进行多卡的dispatch、combine执行次数对比
        if card_num == (all_card_num - 1):
            logging.info("5. 开始多卡的dispatch、combine执行次数对比")
            all_card_dis_run_num = []
            all_card_com_run_num = []
            with open("win_all_card_run_num.csv", "r", encoding="gbk") as f_run_num:
                reader = csv.DictReader(f_run_num)
                for row in reader:
                    all_card_dis_run_num.append(int(row['dispatch执行次数']))
                    all_card_com_run_num.append(int(row['combine执行次数']))
            dis_judge = all(x == all_card_dis_run_num[0] for x in all_card_dis_run_num[:all_card_num])
            com_judge = all(y == all_card_com_run_num[0] for y in all_card_com_run_num[:all_card_num])
            if dis_judge and com_judge:
                logging.info("5. 多卡的dispatch、combine的执行次数完全相同")
            else:
                if not dis_judge:
                    logging.warning("5. 多卡对比中,有卡的dispatch执行次数与其他卡不相同")
                if not com_judge:
                    logging.warning("5. 多卡对比中,有卡的combine执行次数与其他卡不相同")
            logging.info("5. 各卡dispatch执行次数:%s", all_card_dis_run_num[:all_card_num])
            logging.info("5. 各卡combine执行次数:%s", all_card_com_run_num[:all_card_num])
            logging.info("5. 各卡dispatch、combine执行次数数据已归档至win_all_card_run_num.csv")