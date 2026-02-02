/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_mat_mul_all_reduce_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace ge;

namespace {
class GroupedMatMulAllReduceInfershape : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "GroupedMatMulAllReduceInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "GroupedMatMulAllReduceInfershape TearDown" << std::endl;
    }
};

template <typename T>
std::string Shape2String(const T& shape) {
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
        oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_0) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);
    gert::StorageShape x_shape_0 = {{1, 11}, {}};
    gert::StorageShape x_shape_1 = {{2, 12}, {}};
    gert::StorageShape x_shape_2 = {{3, 13}, {}};
    gert::StorageShape x_shape_3 = {{4, 14}, {}};
    gert::StorageShape weight_shape_0 = {{11, 71}, {}};
    gert::StorageShape weight_shape_1 = {{12, 72}, {}};
    gert::StorageShape weight_shape_2 = {{13, 73}, {}};
    gert::StorageShape weight_shape_3 = {{14, 74}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};
    gert::StorageShape y_shape_3 = {{}, {}};
    std::vector<gert::Tensor*> input_shape_ref(8);
    input_shape_ref[0] = (gert::Tensor *)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor *)&x_shape_1;
    input_shape_ref[2] = (gert::Tensor *)&x_shape_2;
    input_shape_ref[3] = (gert::Tensor *)&x_shape_3;
    input_shape_ref[4] = (gert::Tensor *)&weight_shape_0;
    input_shape_ref[5] = (gert::Tensor *)&weight_shape_1;
    input_shape_ref[6] = (gert::Tensor *)&weight_shape_2;
    input_shape_ref[7] = (gert::Tensor *)&weight_shape_3;
    std::vector<gert::StorageShape*> output_shape_ref(4);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;
    output_shape_ref[3] = &y_shape_3;
    auto contextHolder = gert::InferShapeContextFaker() 
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(8, 4)
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .IrInstanceNum({4, 4, 0}, {4})
                        .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*output_shape_0), "[1, 71]");
    auto output_shape_1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*output_shape_1), "[2, 72]");
    auto output_shape_2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*output_shape_2), "[3, 73]");
    auto output_shape_3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*output_shape_3), "[4, 74]");
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_0_multi_dim) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape_0 = {{1, 2, 3, 4, 11}, {}};
    gert::StorageShape x_shape_1 = {{2, 3, 4, 5, 12}, {}};
    gert::StorageShape x_shape_2 = {{3, 4, 5, 6, 13}, {}};
    gert::StorageShape x_shape_3 = {{4, 5, 6, 7, 14}, {}};
    gert::StorageShape weight_shape_0 = {{11, 71}, {}};
    gert::StorageShape weight_shape_1 = {{12, 72}, {}};
    gert::StorageShape weight_shape_2 = {{13, 73}, {}};
    gert::StorageShape weight_shape_3 = {{14, 74}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};
    gert::StorageShape y_shape_3 = {{}, {}};

    std::vector<gert::Tensor*> input_shape_ref(8);
    input_shape_ref[0] = (gert::Tensor *)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor *)&x_shape_1;
    input_shape_ref[2] = (gert::Tensor *)&x_shape_2;
    input_shape_ref[3] = (gert::Tensor *)&x_shape_3;
    input_shape_ref[4] = (gert::Tensor *)&weight_shape_0;
    input_shape_ref[5] = (gert::Tensor *)&weight_shape_1;
    input_shape_ref[6] = (gert::Tensor *)&weight_shape_2;
    input_shape_ref[7] = (gert::Tensor *)&weight_shape_3;

    std::vector<gert::StorageShape*> output_shape_ref(4);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;
    output_shape_ref[3] = &y_shape_3;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(8, 4)
                        .IrInstanceNum({4, 4, 0}, {4})
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*output_shape_0), "[1, 2, 3, 4, 71]");

    auto output_shape_1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*output_shape_1), "[2, 3, 4, 5, 72]");

    auto output_shape_2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*output_shape_2), "[3, 4, 5, 6, 73]");

    auto output_shape_3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*output_shape_3), "[4, 5, 6, 7, 74]");
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_1) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape_0 = {{10, 7}, {}};
    gert::StorageShape weight_shape_0 = {{7, 11}, {}};
    gert::StorageShape weight_shape_1 = {{7, 22}, {}};
    gert::StorageShape weight_shape_2 = {{7, 33}, {}};
    gert::StorageShape weight_shape_3 = {{7, 44}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};
    gert::StorageShape y_shape_3 = {{}, {}};

    int64_t value_size = 4;
    size_t size = static_cast<size_t>(value_size) * sizeof(int64_t);
    int64_t* data_int64 = new int64_t[4];
    data_int64[0] = 1;
    data_int64[1] = 3;
    data_int64[2] = 6;
    data_int64[3] = 10;
    uint8_t* data = reinterpret_cast<uint8_t*>(data_int64);

    ge::DataType const_dtype = ge::DT_INT64;
    uint8_t* group_list_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
    auto input_tensor = reinterpret_cast<gert::Tensor*>(group_list_tensor_holder);
    std::memcpy(input_tensor + 1, data, size);
    gert::Tensor tensor({{value_size}, {value_size}},       // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        const_dtype,                        // dtype
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> input_shape_ref(6);
    input_shape_ref[0] = (gert::Tensor*)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor*)&weight_shape_0;
    input_shape_ref[2] = (gert::Tensor*)&weight_shape_1;
    input_shape_ref[3] = (gert::Tensor*)&weight_shape_2;
    input_shape_ref[4] = (gert::Tensor*)&weight_shape_3;
    input_shape_ref[5] = (gert::Tensor*)group_list_tensor_holder;

    std::vector<gert::StorageShape*> output_shape_ref(4);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;
    output_shape_ref[3] = &y_shape_3;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(6, 4)
                        .IrInstanceNum({1, 4, 0}, {4})
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(1))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, (gert::InferShapeContext*) nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*output_shape_0), "[1, 11]");

    auto output_shape_1 = context->GetOutputShape(1);
    EXPECT_EQ(Shape2String(*output_shape_1), "[2, 22]");

    auto output_shape_2 = context->GetOutputShape(2);
    EXPECT_EQ(Shape2String(*output_shape_2), "[3, 33]");

    auto output_shape_3 = context->GetOutputShape(3);
    EXPECT_EQ(Shape2String(*output_shape_3), "[4, 44]");

    delete[] data_int64;
    delete[] group_list_tensor_holder;
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_1_non_increasing_group_list) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape_0 = {{10, 7}, {}};
    gert::StorageShape weight_shape_0 = {{7, 11}, {}};
    gert::StorageShape weight_shape_1 = {{7, 22}, {}};
    gert::StorageShape weight_shape_2 = {{7, 33}, {}};
    gert::StorageShape weight_shape_3 = {{7, 44}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};
    gert::StorageShape y_shape_1 = {{}, {}};
    gert::StorageShape y_shape_2 = {{}, {}};
    gert::StorageShape y_shape_3 = {{}, {}};

    int64_t value_size = 4;
    size_t size = static_cast<size_t>(value_size) * sizeof(int64_t);
    int64_t* data_int64 = new int64_t[4];
    data_int64[0] = 10;
    data_int64[1] = 6;
    data_int64[2] = 3;
    data_int64[3] = 1;
    uint8_t* data = reinterpret_cast<uint8_t*>(data_int64);

    ge::DataType const_dtype = ge::DT_INT64;
    uint8_t* group_list_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
    auto input_tensor = reinterpret_cast<gert::Tensor*>(group_list_tensor_holder);
    std::memcpy(input_tensor + 1, data, size);
    gert::Tensor tensor({{value_size}, {value_size}},       // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        const_dtype,                        // dtype
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> input_shape_ref(6);
    input_shape_ref[0] = (gert::Tensor*)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor*)&weight_shape_0;
    input_shape_ref[2] = (gert::Tensor*)&weight_shape_1;
    input_shape_ref[3] = (gert::Tensor*)&weight_shape_2;
    input_shape_ref[4] = (gert::Tensor*)&weight_shape_3;
    input_shape_ref[5] = (gert::Tensor*)group_list_tensor_holder;

    std::vector<gert::StorageShape*> output_shape_ref(4);
    output_shape_ref[0] = &y_shape_0;
    output_shape_ref[1] = &y_shape_1;
    output_shape_ref[2] = &y_shape_2;
    output_shape_ref[3] = &y_shape_3;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(6, 4)
                        .IrInstanceNum({1, 4, 0}, {4})
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(1))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(infer_shape_func(context), ge::GRAPH_SUCCESS);

    delete[] data_int64;
    delete[] group_list_tensor_holder;
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_2) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape_0 = {{1, 11}, {}};
    gert::StorageShape x_shape_1 = {{2, 12}, {}};
    gert::StorageShape x_shape_2 = {{3, 13}, {}};
    gert::StorageShape x_shape_3 = {{4, 14}, {}};
    gert::StorageShape weight_shape_0 = {{11, 10}, {}};
    gert::StorageShape weight_shape_1 = {{12, 10}, {}};
    gert::StorageShape weight_shape_2 = {{13, 10}, {}};
    gert::StorageShape weight_shape_3 = {{14, 10}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};

    std::vector<gert::Tensor*> input_shape_ref(8);
    input_shape_ref[0] = (gert::Tensor*)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor*)&x_shape_1;
    input_shape_ref[2] = (gert::Tensor*)&x_shape_2;
    input_shape_ref[3] = (gert::Tensor*)&x_shape_3;
    input_shape_ref[4] = (gert::Tensor*)&weight_shape_0;
    input_shape_ref[5] = (gert::Tensor*)&weight_shape_1;
    input_shape_ref[6] = (gert::Tensor*)&weight_shape_2;
    input_shape_ref[7] = (gert::Tensor*)&weight_shape_3;

    std::vector<gert::StorageShape*> output_shape_ref(1);
    output_shape_ref[0] = &y_shape_0;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(8, 1)
                        .IrInstanceNum({4, 4, 0}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(2))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*output_shape_0), "[10, 10]");
}

TEST_F(GroupedMatMulAllReduceInfershape, grouped_mat_mul_all_reduce_infershape_test_runtime_3) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape_0 = {{10, 7}, {}};
    gert::StorageShape weight_shape_0 = {{7, 10}, {}};
    gert::StorageShape weight_shape_1 = {{7, 10}, {}};
    gert::StorageShape weight_shape_2 = {{7, 10}, {}};
    gert::StorageShape weight_shape_3 = {{7, 10}, {}};
    gert::StorageShape y_shape_0 = {{}, {}};

    int64_t value_size = 4;
    size_t size = static_cast<size_t>(value_size) * sizeof(int64_t);
    int64_t* data_int64 = new int64_t[4];
    data_int64[0] = 1;
    data_int64[1] = 3;
    data_int64[2] = 6;
    data_int64[3] = 10;
    uint8_t* data = reinterpret_cast<uint8_t*>(data_int64);

    ge::DataType const_dtype = ge::DT_INT64;
    uint8_t* group_list_tensor_holder = new uint8_t[sizeof(gert::Tensor) + size];
    auto input_tensor = reinterpret_cast<gert::Tensor*>(group_list_tensor_holder);
    std::memcpy(input_tensor + 1, data, size);
    gert::Tensor tensor({{value_size}, {value_size}},       // shape
                        {ge::FORMAT_ND, ge::FORMAT_ND, {}}, // format
                        gert::kFollowing,                   // placement
                        const_dtype,                        // dtype
                        nullptr);
    std::memcpy(input_tensor, &tensor, sizeof(gert::Tensor));

    std::vector<gert::Tensor*> input_shape_ref(6);
    input_shape_ref[0] = (gert::Tensor*)&x_shape_0;
    input_shape_ref[1] = (gert::Tensor*)&weight_shape_0;
    input_shape_ref[2] = (gert::Tensor*)&weight_shape_1;
    input_shape_ref[3] = (gert::Tensor*)&weight_shape_2;
    input_shape_ref[4] = (gert::Tensor*)&weight_shape_3;
    input_shape_ref[5] = (gert::Tensor*)group_list_tensor_holder;

    std::vector<gert::StorageShape*> output_shape_ref(1);
    output_shape_ref[0] = &y_shape_0;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 4, 0}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes(output_shape_ref)
                        .Attr("splitItem", int64_t(3))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    auto output_shape_0 = context->GetOutputShape(0);
    EXPECT_EQ(Shape2String(*output_shape_0), "[10, 10]");

    delete[] data_int64;
    delete[] group_list_tensor_holder;
}

TEST_F(GroupedMatMulAllReduceInfershape, infer_shape_for_2dim) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{32, 64}, {4, 2, 16, 16}};
    gert::StorageShape weight_shape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape groupList_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<gert::Tensor*> input_shape_ref(4);
    input_shape_ref[0] = (gert::Tensor *)&x_shape;
    input_shape_ref[1] = (gert::Tensor *)&weight_shape;
    input_shape_ref[2] = (gert::Tensor *)&bias_shape;
    input_shape_ref[3] = (gert::Tensor *)&groupList_shape;
    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(4, 1)
                        .IrInstanceNum({1, 1, 1}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes({&output_shape})
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[32, 128]");
}
TEST_F(GroupedMatMulAllReduceInfershape, infer_shape_for_3dim) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{4, 8, 64}, {4, 2, 16, 16}};
    gert::StorageShape weight_shape = {{64, 128}, {8, 4, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape groupList_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<gert::Tensor*> input_shape_ref(4);
    input_shape_ref[0] = (gert::Tensor *)&x_shape;
    input_shape_ref[1] = (gert::Tensor *)&weight_shape;
    input_shape_ref[2] = (gert::Tensor *)&bias_shape;
    input_shape_ref[3] = (gert::Tensor *)&groupList_shape;

    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(4, 1)
                        .IrInstanceNum({1, 1, 1}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes({&output_shape})
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[4, 8, 128]");
}

TEST_F(GroupedMatMulAllReduceInfershape, infer_shape_for_invalid_k) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{32, 8}, {4, 2, 16, 16}};
    gert::StorageShape weight_shape = {{64, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape groupList_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<gert::Tensor*> input_shape_ref(4);
    input_shape_ref[0] = (gert::Tensor *)&x_shape;
    input_shape_ref[1] = (gert::Tensor *)&weight_shape;
    input_shape_ref[2] = (gert::Tensor *)&bias_shape;
    input_shape_ref[3] = (gert::Tensor *)&groupList_shape;
    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(4, 1)
                        .IrInstanceNum({1, 1, 1}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes({&output_shape})
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();

    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

TEST_F(GroupedMatMulAllReduceInfershape, infer_shape_for_zero_k) {
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto infer_shape_func = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x_shape = {{32, 0}, {4, 2, 16, 16}};
    gert::StorageShape weight_shape = {{0, 128}, {4, 2, 16, 16}};
    gert::StorageShape bias_shape = {{128}, {128}};
    gert::StorageShape groupList_shape = {{}, {}};
    gert::StorageShape output_shape = {{}, {}};
    std::vector<gert::Tensor*> input_shape_ref(4);
    input_shape_ref[0] = (gert::Tensor *)&x_shape;
    input_shape_ref[1] = (gert::Tensor *)&weight_shape;
    input_shape_ref[2] = (gert::Tensor *)&bias_shape;
    input_shape_ref[3] = (gert::Tensor *)&groupList_shape;
    auto contextHolder = gert::InferShapeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(4, 1)
                        .IrInstanceNum({1, 1, 1}, {1})
                        .InputTensors(input_shape_ref)
                        .OutputShapes({&output_shape})
                        .Attr("splitItem", int64_t(0))
                        .Attr("group", AscendString("group"))
                        .Attr("reduceOp", AscendString("sum"))
                        .Attr("commTurn", int64_t(0))
                        .Build();
    auto context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);
    auto output = context->GetOutputShape(0);
    ASSERT_EQ(Shape2String(*output), "[32, 128]");
}

TEST_F(GroupedMatMulAllReduceInfershape, infer_dtype) {
    ge::DataType x1_dtype = ge::DT_FLOAT16;
    ge::DataType x2_dtype = ge::DT_FLOAT16;
    ge::DataType y_dtype = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1}, {1})
                        .NodeInputTd(0, x1_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, x2_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeAttrs({{"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                    {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                    {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                                    {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                        .InputDataTypes({&x1_dtype, &x2_dtype})
                        .OutputDataTypes({&y_dtype})
                        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(GroupedMatMulAllReduceInfershape, infer_dtype_test_runtime_2) {
    ge::DataType x_dtype_0 = ge::DT_FLOAT16;
    ge::DataType x_dtype_1 = ge::DT_FLOAT16;
    ge::DataType x_dtype_2 = ge::DT_FLOAT16;
    ge::DataType x_dtype_3 = ge::DT_FLOAT16;
    ge::DataType weight_dtype_0 = ge::DT_FLOAT16;
    ge::DataType weight_dtype_1 = ge::DT_FLOAT16;
    ge::DataType weight_dtype_2 = ge::DT_FLOAT16;
    ge::DataType weight_dtype_3 = ge::DT_FLOAT16;
    ge::DataType y_dtype = ge::DT_UNDEFINED;
    std::vector<void*> input_dtype_ref(8);
    input_dtype_ref[0] = &x_dtype_0;
    input_dtype_ref[1] = &x_dtype_1;
    input_dtype_ref[2] = &x_dtype_2;
    input_dtype_ref[3] = &x_dtype_3;
    input_dtype_ref[4] = &weight_dtype_0;
    input_dtype_ref[5] = &weight_dtype_1;
    input_dtype_ref[6] = &weight_dtype_2;
    input_dtype_ref[7] = &weight_dtype_3;

    auto contextHolder = gert::InferDataTypeContextFaker()
                        .SetOpType("GroupedMatMulAllReduce")
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1}, {1})
                        .NodeInputTd(0, x_dtype_0, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, x_dtype_1, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, x_dtype_2, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, x_dtype_3, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, weight_dtype_0, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, weight_dtype_1, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(6, weight_dtype_2, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(7, weight_dtype_3, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeAttrs({{"splitItem", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                    {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
                                    {"reduceOp", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
                                    {"commTurn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}})
                        .InputDataTypes(input_dtype_ref)
                        .OutputDataTypes({&y_dtype})
                        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("GroupedMatMulAllReduce")->infer_datatype;

    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
}
}