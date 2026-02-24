// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/// @brief Traits class for depthwise convolution forward pass (C=1, K=1 per group).
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
    // Must satisfy: LdsStride - LdsTileW >= PadW (for safe right padding clear)
    // This ensures HorizontalPaddingVector (size=PadW) writes won't overflow into next row
    // when data_width = LdsTileW (worst case for middle tiles)
    static constexpr index_t LdsStrideBase = integer_least_multiple(LdsTileW, InVectorSize);
    static constexpr index_t LdsStrideMin  = LdsTileW + PadW;  // minimum to avoid overflow
    static constexpr index_t LdsStride = (LdsStrideBase >= LdsStrideMin)
        ? LdsStrideBase
        : integer_least_multiple(LdsStrideMin, InVectorSize);

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

// TODO: Future refactoring — split DepthwiseConvFwdTraits into Shape + FilterParams + Traits
// to align with ck_tile conventions (see TileGemmShape/TileGemmTraits pattern).
// This would reduce the 20 template parameters and improve reusability across configurations.

} // namespace ck_tile

