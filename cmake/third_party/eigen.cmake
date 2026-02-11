# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if (IS_DIRECTORY "${CANN_3RD_LIB_PATH}/eigen-5.0.0")
    message(STATUS "Eigen path found in cache: ${CANN_3RD_LIB_PATH}/eigen-5.0.0")
    set(REQ_URL "${CANN_3RD_LIB_PATH}/eigen-5.0.0")
elseif (IS_DIRECTORY "${CANN_3RD_LIB_PATH}/eigen")
    message(STATUS "Eigen path found in cache: ${CANN_3RD_LIB_PATH}/eigen")
    set(REQ_URL "${CANN_3RD_LIB_PATH}/eigen")
else()
    message("The eigen package needs to be downloaded.")
    set(REQ_URL "https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0/eigen-5.0.0.tar.gz")
endif()

include(ExternalProject)
ExternalProject_Add(external_eigen_transformer
        URL               ${REQ_URL}
        DOWNLOAD_DIR      download/eigen
        PREFIX            third_party
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
)

ExternalProject_Get_Property(external_eigen_transformer SOURCE_DIR)

add_library(EigenTransformer INTERFACE)
target_compile_options(EigenTransformer INTERFACE -w)

set_target_properties(EigenTransformer PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}"
)
add_dependencies(EigenTransformer external_eigen_transformer)

add_library(Eigen3::EigenTransformer ALIAS EigenTransformer)