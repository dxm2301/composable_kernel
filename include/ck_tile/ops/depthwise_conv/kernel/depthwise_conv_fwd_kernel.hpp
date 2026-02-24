// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/depthwise_conv/kernel/depthwise_conv_fwd_traits.hpp"

namespace ck_tile {

/// @brief Host-side arguments for depthwise convolution forward pass.
struct DepthwiseConvFwdHostArgs
{
    const void* p_in;   // Input tensor pointer
    const void* p_wei;  // Weight tensor pointer
    void* p_out;        // Output tensor pointer

    // Tensor dimensions [G, N, C, H, W] for input/output, [G, K, C, Y, X] for weight
    index_t G;  // Number of groups
    index_t N;  // Batch size
    index_t C;  // Input channels per group (should be 1 for depthwise)
    index_t K;  // Output channels per group (should be 1 for depthwise)

    // Spatial dimensions
    index_t Hi;  // Input height
    index_t Wi;  // Input width
    index_t Ho;  // Output height
    index_t Wo;  // Output width
    index_t Y;   // Filter height
    index_t X;   // Filter width

    // Convolution parameters
    index_t stride_h;
    index_t stride_w;
    index_t dilation_h;
    index_t dilation_w;
    index_t pad_h;
    index_t pad_w;

    // Tensor strides (for GNCHW/GKCYX layout, matching original CK)
    std::array<index_t, 5> in_strides;   // [g_stride, n_stride, c_stride, h_stride, w_stride]
    std::array<index_t, 5> wei_strides;  // [g_stride, k_stride, c_stride, y_stride, x_stride]
    std::array<index_t, 5> out_strides;  // [g_stride, n_stride, k_stride, h_stride, w_stride]
};

/// @brief Device-side kernel arguments for depthwise convolution.
template <typename Traits_>
struct DepthwiseConvFwdKernelArgs
{
    using Traits      = Traits_;
    using InDataType  = typename Traits::InDataType;
    using WeiDataType = typename Traits::WeiDataType;
    using OutDataType = typename Traits::OutDataType;

    // Pointers
    const InDataType* p_in;
    const WeiDataType* p_wei;
    OutDataType* p_out;

    // Dimensions
    index_t G;
    index_t N;
    index_t Hi;
    index_t Wi;
    index_t Ho;
    index_t Wo;

    // Strides
    index_t in_g_stride;
    index_t in_n_stride;
    index_t in_h_stride;
    index_t in_w_stride;

    index_t wei_g_stride;
    index_t wei_y_stride;
    index_t wei_x_stride;

    index_t out_g_stride;
    index_t out_n_stride;
    index_t out_h_stride;
    index_t out_w_stride;
};

/// @brief Depthwise convolution forward kernel.
template <typename Traits_, typename Pipeline_>
struct DepthwiseConvFwdKernel
{
    using Traits      = Traits_;
    using Pipeline    = Pipeline_;
    using InDataType  = typename Traits::InDataType;
    using WeiDataType = typename Traits::WeiDataType;
    using AccDataType = typename Traits::AccDataType;
    using OutDataType = typename Traits::OutDataType;
    using KernelArgs  = DepthwiseConvFwdKernelArgs<Traits>;

    // Tile configuration
    static constexpr index_t BlockSize   = Traits::BlockSize;
    static constexpr index_t kBlockSize  = BlockSize;  // Alias for make_kernel compatibility
    static constexpr index_t TileOutH    = Traits::TileOutH;
    static constexpr index_t TileOutW    = Traits::TileOutW;
    static constexpr index_t NBatch      = Traits::NBatch;
    static constexpr index_t TilePerWave = Traits::TilePerWave;

    // LDS size
    static constexpr index_t LdsSize = Traits::LdsSize;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return LdsSize; }

    CK_TILE_HOST_DEVICE static constexpr auto BlockSize_() { return dim3(BlockSize); }

    // Grid layout: grid.x = G, grid.y = ceil(N / NBatch)
    CK_TILE_HOST static auto GridSize(index_t G, index_t N)
    {
        const index_t num_batch_groups = integer_divide_ceil(N, NBatch);
        return dim3(G, num_batch_groups, 1);
    }

    CK_TILE_HOST static KernelArgs MakeKernelArgs(const DepthwiseConvFwdHostArgs& args)
    {
        KernelArgs kargs;

        kargs.p_in  = static_cast<const InDataType*>(args.p_in);
        kargs.p_wei = static_cast<const WeiDataType*>(args.p_wei);
        kargs.p_out = static_cast<OutDataType*>(args.p_out);

        kargs.G  = args.G;
        kargs.N  = args.N;
        kargs.Hi = args.Hi;
        kargs.Wi = args.Wi;
        kargs.Ho = args.Ho;
        kargs.Wo = args.Wo;

        // Input strides (GNCHW layout: [g_stride, n_stride, c_stride, h_stride, w_stride])
        kargs.in_g_stride = args.in_strides[0];
        kargs.in_n_stride = args.in_strides[1];
        // c_stride = args.in_strides[2], but C=1 for depthwise so not needed
        kargs.in_h_stride = args.in_strides[3];
        kargs.in_w_stride = args.in_strides[4];

        // Weight strides (GKCYX layout: [g_stride, k_stride, c_stride, y_stride, x_stride])
        kargs.wei_g_stride = args.wei_strides[0];
        // k_stride = args.wei_strides[1], but K=1 for depthwise so not needed
        // c_stride = args.wei_strides[2], but C=1 for depthwise so not needed
        kargs.wei_y_stride = args.wei_strides[3];
        kargs.wei_x_stride = args.wei_strides[4];

        // Output strides (GNKHW layout: [g_stride, n_stride, k_stride, h_stride, w_stride])
        kargs.out_g_stride = args.out_strides[0];
        kargs.out_n_stride = args.out_strides[1];
        // k_stride = args.out_strides[2], but K=1 for depthwise so not needed
        kargs.out_h_stride = args.out_strides[3];
        kargs.out_w_stride = args.out_strides[4];

        return kargs;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const DepthwiseConvFwdHostArgs& args)
    {
        // Check depthwise constraint: C=1, K=1
        if(args.C != 1 || args.K != 1)
        {
            return false;
        }

        // Check filter size matches traits
        if(args.Y != Traits::FilterH || args.X != Traits::FilterW)
        {
            return false;
        }

        // Check stride matches traits
        if(args.stride_h != Traits::StrideH || args.stride_w != Traits::StrideW)
        {
            return false;
        }

        // Check dilation matches traits
        if(args.dilation_h != Traits::DilationH || args.dilation_w != Traits::DilationW)
        {
            return false;
        }

        // Check padding matches traits (symmetric padding required)
        if(args.pad_h != Traits::PadH || args.pad_w != Traits::PadW)
        {
            return false;
        }

        // Check batch size is divisible by NBatch
        if(args.N % NBatch != 0)
        {
            return false;
        }

        // Check NBatch is divisible by TilePerWave for optimal work distribution
        constexpr index_t tile_per_wave = Traits::TilePerWave;
        if(NBatch % tile_per_wave != 0)
        {
            return false;
        }

        // Check LDS size doesn't exceed hardware limit (64KB for most AMD GPUs)
        constexpr index_t max_lds_size = 64 * 1024; // 64KB
        if(Traits::LdsSize > max_lds_size)
        {
            return false;
        }

        // When TilePerWave != 1, load_global_to_lds_with_padding ignores col_offset (global_w_start).
        // This means it can only handle cases where the entire image fits in one tile.
        // If the image is larger than the tile, we need multiple tiles with different global_w_start,
        // which is not supported by load_global_to_lds_with_padding.
        // So we must reject cases where TilePerWave != 1 and image > tile size.
        if constexpr(Traits::TilePerWave != 1)
        {
            if(args.Ho > Traits::TileOutH || args.Wo > Traits::TileOutW)
            {
                return false;
            }
        }

        // Reject cases where input spatial size is smaller than kernel size.
        // These edge cases can cause incorrect results due to boundary handling issues.
        if(args.Hi < args.Y || args.Wi < args.X)
        {
            return false;
        }

        return true;
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        // Get block indices
        const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t batch_group = __builtin_amdgcn_readfirstlane(blockIdx.y);

        // Calculate base pointers for this block
        const auto* p_in_base =
            kargs.p_in + static_cast<long_index_t>(g_idx) * kargs.in_g_stride +
            static_cast<long_index_t>(batch_group * NBatch) * kargs.in_n_stride;

        const auto* p_wei_base =
            kargs.p_wei + static_cast<long_index_t>(g_idx) * kargs.wei_g_stride;

        auto* p_out_base = kargs.p_out + static_cast<long_index_t>(g_idx) * kargs.out_g_stride +
                           static_cast<long_index_t>(batch_group * NBatch) * kargs.out_n_stride;

        // Allocate LDS
        __shared__ char smem[GetSmemSize()];

        // Run the pipeline
        Pipeline{}(p_in_base,
                   p_wei_base,
                   p_out_base,
                   smem,
                   kargs.Hi,
                   kargs.Wi,
                   kargs.Ho,
                   kargs.Wo,
                   kargs.in_h_stride,
                   kargs.in_w_stride,
                   kargs.in_n_stride,
                   kargs.wei_y_stride,
                   kargs.wei_x_stride,
                   kargs.out_h_stride,
                   kargs.out_w_stride,
                   kargs.out_n_stride);
    }
};

} // namespace ck_tile

