// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/**
 * @brief Shape configuration for depthwise convolution forward pass.
 *
 * This structure defines the tile shape and work distribution for depthwise
 * convolution. Unlike GEMM which has M×N×K dimensions, depthwise conv operates
 * in spatial domain with H×W dimensions.
 *
 * @tparam TileOutH_    Output tile height
 * @tparam TileOutW_    Output tile width
 * @tparam SubTileH_    Per-thread output height
 * @tparam SubTileW_    Per-thread output width
 * @tparam NBatch_      Number of batches per block
 */
template <index_t TileOutH_,
          index_t TileOutW_,
          index_t SubTileH_,
          index_t SubTileW_,
          index_t NBatch_>
struct DepthwiseConvFwdShape
{
    // Output tile dimensions (block-level)
    static constexpr index_t TileOutH = TileOutH_;
    static constexpr index_t TileOutW = TileOutW_;

    // Per-thread output dimensions
    static constexpr index_t SubTileH = SubTileH_;
    static constexpr index_t SubTileW = SubTileW_;

    // Batch processing
    static constexpr index_t NBatch = NBatch_;

    // Work distribution within a wave
    static constexpr index_t WaveSize      = 64;
    static constexpr index_t HRepeats      = integer_divide_ceil(TileOutH, SubTileH);
    static constexpr index_t WRepeats      = integer_divide_ceil(TileOutW, SubTileW);
    static constexpr index_t TotalSubTiles = HRepeats * WRepeats;
    static constexpr index_t TilePerWave   = WaveSize / TotalSubTiles;
    static constexpr index_t ThreadPerTile = WaveSize / TilePerWave;

    // Validation
    static_assert(TotalSubTiles <= WaveSize, "TotalSubTiles must fit in a wave");
    static_assert(NBatch % TilePerWave == 0, "NBatch must be divisible by TilePerWave");
};

/**
 * @brief Filter parameter configuration.
 *
 * @tparam FilterH_     Filter height
 * @tparam FilterW_     Filter width
 * @tparam StrideH_     Vertical stride
 * @tparam StrideW_     Horizontal stride
 * @tparam DilationH_   Vertical dilation
 * @tparam DilationW_   Horizontal dilation
 * @tparam PadH_        Vertical padding
 * @tparam PadW_        Horizontal padding
 */
template <index_t FilterH_,
          index_t FilterW_,
          index_t StrideH_,
          index_t StrideW_,
          index_t DilationH_,
          index_t DilationW_,
          index_t PadH_,
          index_t PadW_>
struct DepthwiseConvFilterParams
{
    static constexpr index_t FilterH   = FilterH_;
    static constexpr index_t FilterW   = FilterW_;
    static constexpr index_t StrideH   = StrideH_;
    static constexpr index_t StrideW   = StrideW_;
    static constexpr index_t DilationH = DilationH_;
    static constexpr index_t DilationW = DilationW_;
    static constexpr index_t PadH      = PadH_;
    static constexpr index_t PadW      = PadW_;

    // Effective filter size with dilation
    static constexpr index_t EffectiveFilterH = (FilterH - 1) * DilationH + 1;
    static constexpr index_t EffectiveFilterW = (FilterW - 1) * DilationW + 1;

    // Calculate output size from input size
    CK_TILE_HOST_DEVICE static constexpr index_t GetOutputH(index_t input_h)
    {
        return (input_h + 2 * PadH - EffectiveFilterH) / StrideH + 1;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetOutputW(index_t input_w)
    {
        return (input_w + 2 * PadW - EffectiveFilterW) / StrideW + 1;
    }

    // Calculate required input size for given output size
    CK_TILE_HOST_DEVICE static constexpr index_t GetRequiredInputH(index_t output_h)
    {
        return output_h * StrideH + EffectiveFilterH - 1;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetRequiredInputW(index_t output_w)
    {
        return output_w * StrideW + EffectiveFilterW - 1;
    }
};

/**
 * @brief Common filter configurations.
 */
using FilterParams_3x3_S1_P1 = DepthwiseConvFilterParams<3, 3, 1, 1, 1, 1, 1, 1>;
using FilterParams_3x3_S2_P1 = DepthwiseConvFilterParams<3, 3, 2, 2, 1, 1, 1, 1>;
using FilterParams_5x5_S1_P2 = DepthwiseConvFilterParams<5, 5, 1, 1, 1, 1, 2, 2>;
using FilterParams_5x5_S2_P2 = DepthwiseConvFilterParams<5, 5, 2, 2, 1, 1, 2, 2>;
using FilterParams_7x7_S1_P3 = DepthwiseConvFilterParams<7, 7, 1, 1, 1, 1, 3, 3>;
using FilterParams_7x7_S2_P3 = DepthwiseConvFilterParams<7, 7, 2, 2, 1, 1, 3, 3>;
using FilterParams_9x9_S1_P4 = DepthwiseConvFilterParams<9, 9, 1, 1, 1, 1, 4, 4>;
using FilterParams_9x9_S2_P4 = DepthwiseConvFilterParams<9, 9, 2, 2, 1, 1, 4, 4>;

} // namespace ck_tile

