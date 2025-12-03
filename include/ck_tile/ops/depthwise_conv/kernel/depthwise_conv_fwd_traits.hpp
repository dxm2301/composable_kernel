// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/**
 * @brief Traits class for depthwise convolution forward pass.
 *
 * This class defines compile-time constants and type aliases for the depthwise
 * convolution operation. Unlike grouped convolution which uses implicit GEMM,
 * depthwise convolution has C=1 and K=1 per group, so it's computed directly
 * in spatial domain.
 *
 * @tparam InDataType_      Input tensor data type (e.g., fp16_t, float)
 * @tparam WeiDataType_     Weight tensor data type
 * @tparam AccDataType_     Accumulation data type (typically float)
 * @tparam OutDataType_     Output tensor data type
 * @tparam BlockSize_       Number of threads per block (typically 64 or 256)
 * @tparam TileH_           Output tile height
 * @tparam TileW_           Output tile width
 * @tparam FilterH_         Convolution kernel height
 * @tparam FilterW_         Convolution kernel width
 * @tparam StrideH_         Vertical stride
 * @tparam StrideW_         Horizontal stride
 * @tparam DilationH_       Vertical dilation
 * @tparam DilationW_       Horizontal dilation
 * @tparam PadH_            Vertical padding (same for top/bottom)
 * @tparam PadW_            Horizontal padding (same for left/right)
 * @tparam NBatch_          Number of batches processed per block
 * @tparam SubTileH_        Sub-tile height (per-thread output)
 * @tparam SubTileW_        Sub-tile width (per-thread output)
 * @tparam InVectorSize_    Input vector load width
 * @tparam OutVectorSize_   Output vector store width
 */
template <typename InDataType_,
          typename WeiDataType_,
          typename AccDataType_,
          typename OutDataType_,
          index_t BlockSize_,
          index_t TileH_,
          index_t TileW_,
          index_t FilterH_,
          index_t FilterW_,
          index_t StrideH_,
          index_t StrideW_,
          index_t DilationH_,
          index_t DilationW_,
          index_t PadH_,
          index_t PadW_,
          index_t NBatch_,
          index_t SubTileH_,
          index_t SubTileW_,
          index_t InVectorSize_,
          index_t OutVectorSize_>
struct DepthwiseConvFwdTraits
{
    // Data types
    using InDataType  = InDataType_;
    using WeiDataType = WeiDataType_;
    using AccDataType = AccDataType_;
    using OutDataType = OutDataType_;

    // Spatial dimensions
    static constexpr index_t NDimSpatial = 2;

    // Block configuration
    static constexpr index_t BlockSize = BlockSize_;
    static constexpr index_t WaveSize  = 64;

    // Output tile dimensions (in output space)
    static constexpr index_t TileOutH = TileH_;
    static constexpr index_t TileOutW = TileW_;

    // Input tile dimensions (derived from output tile and stride)
    static constexpr index_t TileInH = TileOutH * StrideH_;
    static constexpr index_t TileInW = TileOutW * StrideW_;

    // Filter dimensions
    static constexpr index_t FilterH = FilterH_;
    static constexpr index_t FilterW = FilterW_;

    // Convolution parameters
    static constexpr index_t StrideH   = StrideH_;
    static constexpr index_t StrideW   = StrideW_;
    static constexpr index_t DilationH = DilationH_;
    static constexpr index_t DilationW = DilationW_;
    static constexpr index_t PadH      = PadH_;
    static constexpr index_t PadW      = PadW_;

    // LDS tile dimensions (input tile + padding)
    static constexpr index_t LdsTileH = TileInH + 2 * PadH;
    static constexpr index_t LdsTileW = TileInW + 2 * PadW;

    // Batch processing
    static constexpr index_t NBatch = NBatch_;

    // Sub-tile dimensions (per-thread output)
    static constexpr index_t SubTileH = SubTileH_;
    static constexpr index_t SubTileW = SubTileW_;

    // Vectorization
    static constexpr index_t InVectorSize  = InVectorSize_;
    static constexpr index_t OutVectorSize = OutVectorSize_;
    static constexpr index_t WeiVectorSize = 2; // Weight vector size for inner product

    // Derived constants for work distribution
    static constexpr index_t HRepeats      = integer_divide_ceil(TileOutH, SubTileH);
    static constexpr index_t WRepeats      = integer_divide_ceil(TileOutW, SubTileW);
    static constexpr index_t TotalSubTiles = HRepeats * WRepeats;
    static constexpr index_t TilePerWave   = WaveSize / TotalSubTiles;
    static constexpr index_t ThreadPerTile = WaveSize / TilePerWave;

    // LDS stride (aligned for vector access)
    static constexpr index_t LdsStride = integer_least_multiple(LdsTileW, InVectorSize);

    // LDS size per tile
    static constexpr index_t LdsTileSize = LdsTileH * LdsStride;

    // Total LDS size for input (all tiles in a wave)
    static constexpr index_t LdsInputSize = LdsTileSize * TilePerWave * sizeof(InDataType);

    // Minimum LDS size (output is written directly to global memory)
    static constexpr index_t LdsSize = LdsInputSize;

    // Vector types (using ck_tile's ext_vector_t)
    using InVector  = ext_vector_t<InDataType, InVectorSize>;
    using OutVector = ext_vector_t<OutDataType, OutVectorSize>;
    using WeiVector = ext_vector_t<WeiDataType, WeiVectorSize>;

    // Internal vector sizes (capped at 4 for LDS access)
    static constexpr index_t InVectorSizeInternal  = (InVectorSize < 4) ? InVectorSize : 4;
    static constexpr index_t OutVectorSizeInternal = (OutVectorSize < 4) ? OutVectorSize : 4;

    using InVectorInternal  = ext_vector_t<InDataType, InVectorSizeInternal>;
    using OutVectorInternal = ext_vector_t<OutDataType, OutVectorSizeInternal>;
    using AccVectorInternal = ext_vector_t<AccDataType, OutVectorSizeInternal>;

    // Validation
    static_assert(BlockSize == 64 || BlockSize == 128 || BlockSize == 256,
                  "BlockSize must be 64, 128, or 256");
    static_assert(TotalSubTiles <= WaveSize, "TotalSubTiles must not exceed WaveSize");
    static_assert(DilationH == 1 && DilationW == 1, "Only dilation=1 is supported currently");
    static_assert(FilterH == FilterW, "Only square filters are supported currently");
    static_assert(FilterH % 2 == 1, "Only odd filter sizes are supported (3, 5, 7, 9)");
    // NBatch validation moved to runtime check in IsSupportedArgument
    // NBatch should ideally be divisible by TilePerWave for optimal performance
};

/**
 * @brief Commonly used depthwise conv traits configurations.
 */

// 3x3 kernel, stride 1, fp16
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType>
using DepthwiseConvFwdTraits_3x3_S1 = DepthwiseConvFwdTraits<InDataType,
                                                             WeiDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             64,   // BlockSize
                                                             16,   // TileH
                                                             16,   // TileW
                                                             3,    // FilterH
                                                             3,    // FilterW
                                                             1,    // StrideH
                                                             1,    // StrideW
                                                             1,    // DilationH
                                                             1,    // DilationW
                                                             1,    // PadH
                                                             1,    // PadW
                                                             64,   // NBatch
                                                             4,    // SubTileH
                                                             4,    // SubTileW
                                                             8,    // InVectorSize
                                                             8>;   // OutVectorSize

// 5x5 kernel, stride 1, fp16
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType>
using DepthwiseConvFwdTraits_5x5_S1 = DepthwiseConvFwdTraits<InDataType,
                                                             WeiDataType,
                                                             AccDataType,
                                                             OutDataType,
                                                             64,   // BlockSize
                                                             16,   // TileH
                                                             16,   // TileW
                                                             5,    // FilterH
                                                             5,    // FilterW
                                                             1,    // StrideH
                                                             1,    // StrideW
                                                             1,    // DilationH
                                                             1,    // DilationW
                                                             2,    // PadH
                                                             2,    // PadW
                                                             64,   // NBatch
                                                             4,    // SubTileH
                                                             4,    // SubTileW
                                                             8,    // InVectorSize
                                                             8>;   // OutVectorSize

} // namespace ck_tile

