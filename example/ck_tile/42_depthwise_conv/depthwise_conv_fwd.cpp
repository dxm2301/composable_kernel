// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "depthwise_conv_fwd_invoker.hpp"
#include "run_depthwise_conv_fwd_example.inc"

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string data_type = arg_parser.get_str("prec");

    try
    {
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
        // TODO: add bf16/fp8 dispatch when kernel supports it
        else
        {
            throw std::runtime_error("Unsupported data type: " + data_type);
        }
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Runtime error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
