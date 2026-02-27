// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/depthwise_conv/kernel/depthwise_conv_fwd_traits.hpp"

namespace ck_tile {

/// @brief Host-side arguments for depthwise convolution forward pass.
struct DepthwiseConvFwdHostArgs
{
    const void* p_in;
    const void* p_wei;
    void* p_out;

    // Layout — Input: [G,N,C,Hi,Wi], Weight: [G,K,C,Y,X], Output: [G,N,K,Ho,Wo]
    index_t G;
    index_t N;
    index_t C;
    index_t K;
    index_t Hi;
    index_t Wi;
    index_t Ho;
    index_t Wo;
    index_t Y;
    index_t X;

    index_t stride_h;
    index_t stride_w;
    index_t dilation_h;
    index_t dilation_w;
    index_t pad_h;
    index_t pad_w;

    // Tensor strides, indexed by dimension — In: GNCHW, Wei: GKCYX, Out: GNKHW
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

    const InDataType* p_in;
    const WeiDataType* p_wei;
    OutDataType* p_out;

    index_t G;
    index_t N;
    index_t Hi;
    index_t Wi;
    index_t Ho;
    index_t Wo;

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

    static constexpr index_t BlockSize   = Traits::BlockSize;
    static constexpr index_t kBlockSize  = BlockSize;  // Required by ck_tile::make_kernel
    static constexpr index_t TileOutH    = Traits::TileOutH;
    static constexpr index_t TileOutW    = Traits::TileOutW;
    static constexpr index_t NBatch      = Traits::NBatch;
    static constexpr index_t TilePerWave = Traits::TilePerWave;

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

        // Strides [2] (c_stride / k_stride) skipped: C=K=1 for depthwise
        kargs.in_g_stride = args.in_strides[0];
        kargs.in_n_stride = args.in_strides[1];
        kargs.in_h_stride = args.in_strides[3];
        kargs.in_w_stride = args.in_strides[4];

        kargs.wei_g_stride = args.wei_strides[0];
        kargs.wei_y_stride = args.wei_strides[3];
        kargs.wei_x_stride = args.wei_strides[4];

        kargs.out_g_stride = args.out_strides[0];
        kargs.out_n_stride = args.out_strides[1];
        kargs.out_h_stride = args.out_strides[3];
        kargs.out_w_stride = args.out_strides[4];

        return kargs;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const DepthwiseConvFwdHostArgs& args)
    {
        if(args.C != 1 || args.K != 1)
        {
            return false;
        }

        if(args.Y != Traits::FilterH || args.X != Traits::FilterW)
        {
            return false;
        }

        if(args.stride_h != Traits::StrideH || args.stride_w != Traits::StrideW)
        {
            return false;
        }

        if(args.dilation_h != Traits::DilationH || args.dilation_w != Traits::DilationW)
        {
            return false;
        }

        // Same padding on both sides per dimension
        if(args.pad_h != Traits::PadH || args.pad_w != Traits::PadW)
        {
            return false;
        }

        if(args.N % NBatch != 0)
        {
            return false;
        }

        // Compile-time config filtering (if constexpr to allow invoker to skip invalid instantiations)
        if constexpr(NBatch % Traits::TilePerWave != 0)
        {
            return false;
        }

        if constexpr(Traits::LdsSize > 64 * 1024)
        {
            return false;
        }

        // TilePerWave > 1 requires the entire spatial output to fit in one tile,
        // because the LDS loader does not support per-tile column offsets.
        if constexpr(Traits::TilePerWave != 1)
        {
            if(args.Ho > Traits::TileOutH || args.Wo > Traits::TileOutW)
            {
                return false;
            }
        }

        // Input spatial dims must be >= kernel size to avoid out-of-bound LDS access.
        if(args.Hi < args.Y || args.Wi < args.X)
        {
            return false;
        }

        return true;
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        const index_t g_idx       = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t batch_group = __builtin_amdgcn_readfirstlane(blockIdx.y);

        const auto* p_in_base =
            kargs.p_in + static_cast<long_index_t>(g_idx) * kargs.in_g_stride +
            static_cast<long_index_t>(batch_group * NBatch) * kargs.in_n_stride;

        const auto* p_wei_base =
            kargs.p_wei + static_cast<long_index_t>(g_idx) * kargs.wei_g_stride;

        auto* p_out_base = kargs.p_out + static_cast<long_index_t>(g_idx) * kargs.out_g_stride +
                           static_cast<long_index_t>(batch_group * NBatch) * kargs.out_n_stride;

        __shared__ char smem[GetSmemSize()];

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
