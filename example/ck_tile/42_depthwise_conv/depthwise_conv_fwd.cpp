// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "depthwise_conv_utils.hpp"
#include "depthwise_conv_fwd_invoker.hpp"
#include "run_depthwise_conv_fwd_example.inc"

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");

    if(data_type == "fp32")
    {
        return run_depthwise_conv_fwd_example<float, float, float, float>(argc, argv);
    }
    else if(data_type == "fp16")
    {
        return run_depthwise_conv_fwd_example<ck_tile::half_t,
                                              ck_tile::half_t,
                                              float,
                                              ck_tile::half_t>(argc, argv);
    }
    else
    {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return -1;
    }
}

