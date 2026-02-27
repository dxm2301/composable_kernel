// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/depthwise_conv.hpp"

struct DepthwiseConvParam
{
    ck_tile::index_t num_dim_spatial_ = 2;
    ck_tile::index_t G_               = 1;   // number of groups (= total input channels for depthwise)
    ck_tile::index_t N_               = 1;   // batch size
    ck_tile::index_t C_               = 1;   // input channels per group (always 1 for depthwise)
    ck_tile::index_t K_               = 1;   // output channels per group (always 1 for depthwise)

    std::vector<ck_tile::index_t> input_spatial_lengths_  = {32, 32};
    std::vector<ck_tile::index_t> filter_spatial_lengths_ = {3, 3};
    std::vector<ck_tile::index_t> output_spatial_lengths_ = {32, 32};

    std::vector<ck_tile::index_t> conv_filter_strides_   = {1, 1};
    std::vector<ck_tile::index_t> conv_filter_dilations_ = {1, 1};
    std::vector<ck_tile::index_t> input_left_pads_       = {1, 1};
    std::vector<ck_tile::index_t> input_right_pads_      = {1, 1};

    DepthwiseConvParam() = default;

    DepthwiseConvParam(ck_tile::index_t num_dim_spatial,
                       ck_tile::index_t G,
                       ck_tile::index_t N,
                       ck_tile::index_t C,
                       ck_tile::index_t K,
                       std::vector<ck_tile::index_t> input_spatial_lengths,
                       std::vector<ck_tile::index_t> filter_spatial_lengths,
                       std::vector<ck_tile::index_t> conv_filter_strides,
                       std::vector<ck_tile::index_t> conv_filter_dilations,
                       std::vector<ck_tile::index_t> input_left_pads,
                       std::vector<ck_tile::index_t> input_right_pads)
        : num_dim_spatial_(num_dim_spatial),
          G_(G),
          N_(N),
          C_(C),
          K_(K),
          input_spatial_lengths_(input_spatial_lengths),
          filter_spatial_lengths_(filter_spatial_lengths),
          conv_filter_strides_(conv_filter_strides),
          conv_filter_dilations_(conv_filter_dilations),
          input_left_pads_(input_left_pads),
          input_right_pads_(input_right_pads)
    {
        compute_output_spatial_lengths();
    }

    void compute_output_spatial_lengths()
    {
        output_spatial_lengths_.clear();
        for(ck_tile::index_t i = 0; i < num_dim_spatial_; ++i)
        {
            const ck_tile::index_t dilated_filter =
                conv_filter_dilations_[i] * (filter_spatial_lengths_[i] - 1) + 1;
            const ck_tile::index_t out_len =
                (input_spatial_lengths_[i] + input_left_pads_[i] + input_right_pads_[i] -
                 dilated_filter) /
                    conv_filter_strides_[i] +
                1;
            output_spatial_lengths_.push_back(out_len);
        }
    }

    std::size_t GetFlops() const
    {
        std::size_t flops = static_cast<std::size_t>(N_) * G_ * C_ * K_;
        for(ck_tile::index_t i = 0; i < num_dim_spatial_; ++i)
        {
            flops *= filter_spatial_lengths_[i] * output_spatial_lengths_[i];
        }
        return 2 * flops; // multiply-add
    }

    template <typename InDataType, typename WeiDataType, typename OutDataType>
    std::size_t GetByte() const
    {
        std::size_t input_size = static_cast<std::size_t>(N_) * G_ * C_;
        for(auto len : input_spatial_lengths_)
            input_size *= len;

        std::size_t weight_size = static_cast<std::size_t>(G_) * K_ * C_;
        for(auto len : filter_spatial_lengths_)
            weight_size *= len;

        std::size_t output_size = static_cast<std::size_t>(N_) * G_ * K_;
        for(auto len : output_spatial_lengths_)
            output_size *= len;

        return input_size * sizeof(InDataType) + weight_size * sizeof(WeiDataType) +
               output_size * sizeof(OutDataType);
    }
};

// Strides for [G, N, C/K, H, W]. Used for both input (GNCHW) and output (GNKHW).
// Returns: {g=C*H*W, n=G*C*H*W, c=1, h=W*C, w=C}
std::array<ck_tile::index_t, 5> GetGNCHWStrides(ck_tile::index_t G,
                                                [[maybe_unused]] ck_tile::index_t N,
                                                ck_tile::index_t C,
                                                ck_tile::index_t H,
                                                ck_tile::index_t W)
{
    return {C * H * W, G * C * H * W, 1, W * C, C};
}

// Strides for [G, K, C, Y, X] (weight). K=C=1 for depthwise.
// Returns: {g=C*Y*X, k=G*C*Y*X, c=1, y=X, x=1}
std::array<ck_tile::index_t, 5> GetGKCYXStrides(ck_tile::index_t G,
                                                [[maybe_unused]] ck_tile::index_t K,
                                                ck_tile::index_t C,
                                                ck_tile::index_t Y,
                                                ck_tile::index_t X)
{
    return {C * Y * X, G * C * Y * X, 1, X, 1};
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("prec", "fp32", "data type (fp32, fp16)")
        .insert("v", "1", "verification (0=no, 1=yes)")
        .insert("warmup", "5", "number of warmup iterations")
        .insert("repeat", "20", "number of repeat iterations")
        .insert("G", "128", "number of groups (channels for depthwise)")
        .insert("N", "64", "batch size")
        .insert("H", "56", "input height")
        .insert("W", "56", "input width")
        .insert("Y", "3", "filter height")
        .insert("X", "3", "filter width")
        .insert("stride_h", "1", "stride height")
        .insert("stride_w", "1", "stride width")
        .insert("dilation_h", "1", "dilation height")
        .insert("dilation_w", "1", "dilation width")
        .insert("pad_h", "1", "padding height")
        .insert("pad_w", "1", "padding width")
        .insert("init", "2", "init (0=zero,1=int,2=rand,3=col+w1,4=row+w1,5=col+wspatial)");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}
