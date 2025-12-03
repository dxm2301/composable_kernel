// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

// Depthwise convolution parameters
struct DepthwiseConvParam
{
    ck_tile::index_t num_dim_spatial_ = 2;
    ck_tile::index_t G_               = 1;   // number of groups (= C for depthwise)
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

/**
 * @brief Calculate tensor strides for GNCHW layout (input/output).
 *
 * Layout: [G, N, C, H, W] where C=1 for depthwise
 * This matches the original CK layout from common.hpp
 *
 * Memory layout (from common.hpp make_input_descriptor for 2D):
 *   g_stride = C * H * W
 *   n_stride = G * C * H * W
 *   c_stride = 1  (C is interleaved, but C=1 for depthwise so doesn't matter)
 *   h_stride = W * C
 *   w_stride = C
 */
inline std::array<ck_tile::index_t, 5> GetGNCHWStrides(ck_tile::index_t G,
                                                       [[maybe_unused]] ck_tile::index_t N,
                                                       ck_tile::index_t C,
                                                       ck_tile::index_t H,
                                                       ck_tile::index_t W)
{
    // Strides: [g_stride, n_stride, c_stride, h_stride, w_stride]
    // From common.hpp: GNCHW with C interleaved
    return {C * H * W,          // g_stride
            G * C * H * W,      // n_stride
            1,                  // c_stride (C=1 for depthwise, so this is effectively 1)
            W * C,              // h_stride
            C};                 // w_stride
}

/**
 * @brief Calculate tensor strides for GKCYX layout (weight).
 *
 * Layout: [G, K, C, Y, X] where K=C=1 for depthwise
 * This matches the original CK layout from common.hpp
 *
 * Memory layout (from common.hpp make_weight_descriptor for 2D):
 *   g_stride = C * Y * X
 *   k_stride = G * C * Y * X
 *   c_stride = 1
 *   y_stride = X
 *   x_stride = 1
 */
inline std::array<ck_tile::index_t, 5> GetGKCYXStrides(ck_tile::index_t G,
                                                       [[maybe_unused]] ck_tile::index_t K,
                                                       ck_tile::index_t C,
                                                       ck_tile::index_t Y,
                                                       ck_tile::index_t X)
{
    // Strides: [g_stride, k_stride, c_stride, y_stride, x_stride]
    return {C * Y * X,          // g_stride
            G * C * Y * X,      // k_stride
            1,                  // c_stride
            X,                  // y_stride (note: no C factor here based on original)
            1};                 // x_stride
}

} // namespace ck_tile

// Helper function to create argument parser
inline auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("prec", "fp32", "data type (fp32, fp16)")
        .insert("v", "1", "verification (0=no, 1=yes)")
        .insert("warmup", "5", "number of warmup iterations")
        .insert("repeat", "20", "number of repeat iterations")
        .insert("timer", "gpu", "timer type (gpu, cpu)")
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
