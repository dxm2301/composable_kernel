// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/depthwise_conv/kernel/depthwise_conv_fwd_traits.hpp"

// Debug flags - uncomment to enable debug output
// #define CK_TILE_DEPTHWISE_DEBUG

namespace ck_tile {

// ==================== Inner Product (Hardware-Optimized) ====================
// Uses AMD GPU builtin instructions for maximum performance:
// - __builtin_amdgcn_fdot2: v_dot2_f32_f16 (2x FP16 dot product in single instruction)
// - Direct FMA for float

template <typename T>
CK_TILE_DEVICE void depthwise_inner_product(const T& a, const T& b, float& c)
{
    // FP16x2: use hardware v_dot2_f32_f16 instruction
    if constexpr(std::is_same_v<T, fp16x2_t>)
    {
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
        c = __builtin_amdgcn_fdot2(a, b, c, false);
#else
        c += static_cast<float>(a[0]) * static_cast<float>(b[0]);
        c += static_cast<float>(a[1]) * static_cast<float>(b[1]);
#endif
    }
    // Float scalar
    else if constexpr(std::is_same_v<T, float>)
    {
        c += a * b;
    }
    // Float2: two scalar FMAs
    else if constexpr(sizeof(T) == sizeof(float) * 2 && 
                      std::is_same_v<typename vector_traits<T>::scalar_type, float>)
    {
        c += a[0] * b[0];
        c += a[1] * b[1];
    }
    // Generic 2-element vector fallback
    else
    {
        c += static_cast<float>(a[0]) * static_cast<float>(b[0]);
        c += static_cast<float>(a[1]) * static_cast<float>(b[1]);
    }
}

// ==================== Debug Utilities (Matching Original CK Exactly) ====================

/**
 * @brief Dump LDS contents for debugging.
 * Matches original CK's dump_lds function exactly.
 */
template <typename DataType>
CK_TILE_DEVICE void dump_lds(DataType* p, index_t totalcount, index_t length)
{
    for(index_t i = 0; i < totalcount; i++)
    {
        if(i % length == 0)
        {
            printf("\n [%d]", static_cast<int>(i / length));
        }

        if constexpr(std::is_same_v<DataType, fp16_t> || std::is_same_v<DataType, half_t>)
        {
            printf("%.3f ", static_cast<float>(p[i]));
        }
        else if constexpr(std::is_same_v<DataType, bf16_t>)
        {
            printf("%.3f ", type_convert<float>(p[i]));
        }
        else if constexpr(std::is_same_v<DataType, float>)
        {
            printf("%.3f ", p[i]);
        }
        else if constexpr(std::is_same_v<DataType, int8_t> || std::is_same_v<DataType, uint8_t>)
        {
            printf("%4d ", static_cast<int>(p[i]));
        }
        else
        {
            printf("0x%x ", *reinterpret_cast<const uint16_t*>(&p[i]));
        }
    }
    printf("\n");
}

/**
 * @brief Pipeline for depthwise convolution forward pass.
 *
 * This pipeline implements the core computation logic for depthwise convolution:
 * 1. Load input tile from global memory to LDS (with padding)
 * 2. Load filter weights to registers
 * 3. Perform convolution using circular buffer for filter rows
 * 4. Write output directly to global memory
 *
 * Data flow:
 *   Global Memory (Input) → VGPR → LDS (with padding) → VGPR (circular buffer)
 *   → Compute → VGPR (accumulator) → Global Memory (Output)
 *
 * @tparam Traits_ Traits class defining types and compile-time constants
 */
template <typename Traits_>
struct DepthwiseConvFwdPipeline
{
    using Traits = Traits_;

    // Data types
    using InDataType  = typename Traits::InDataType;
    using WeiDataType = typename Traits::WeiDataType;
    using AccDataType = typename Traits::AccDataType;
    using OutDataType = typename Traits::OutDataType;

    // Vector types
    using InVector         = typename Traits::InVector;
    using OutVector        = typename Traits::OutVector;
    using WeiVector        = typename Traits::WeiVector;
    using InVectorInternal = typename Traits::InVectorInternal;

    // Compile-time constants
    static constexpr index_t BlockSize   = Traits::BlockSize;
    static constexpr index_t WaveSize    = Traits::WaveSize;
    static constexpr index_t TileOutH    = Traits::TileOutH;
    static constexpr index_t TileOutW    = Traits::TileOutW;
    static constexpr index_t TileInH     = Traits::TileInH;
    static constexpr index_t TileInW     = Traits::TileInW;
    static constexpr index_t LdsTileH    = Traits::LdsTileH;
    static constexpr index_t LdsTileW    = Traits::LdsTileW;
    static constexpr index_t LdsStride   = Traits::LdsStride;
    static constexpr index_t LdsTileSize = Traits::LdsTileSize;

    static constexpr index_t FilterH = Traits::FilterH;
    static constexpr index_t FilterW = Traits::FilterW;
    static constexpr index_t StrideH = Traits::StrideH;
    static constexpr index_t StrideW = Traits::StrideW;
    static constexpr index_t PadH    = Traits::PadH;
    static constexpr index_t PadW    = Traits::PadW;

    static constexpr index_t NBatch       = Traits::NBatch;
    static constexpr index_t SubTileH     = Traits::SubTileH;
    static constexpr index_t SubTileW     = Traits::SubTileW;
    static constexpr index_t HRepeats     = Traits::HRepeats;
    static constexpr index_t WRepeats     = Traits::WRepeats;
    static constexpr index_t TilePerWave  = Traits::TilePerWave;
    static constexpr index_t ThreadPerTile = Traits::ThreadPerTile;

    static constexpr index_t InVectorSize         = Traits::InVectorSize;
    static constexpr index_t OutVectorSize        = Traits::OutVectorSize;
    static constexpr index_t WeiVectorSize        = Traits::WeiVectorSize;
    static constexpr index_t InVectorSizeInternal = Traits::InVectorSizeInternal;

    // Derived constants
    static constexpr index_t FilterXPack = integer_divide_ceil(FilterW, WeiVectorSize);
    static constexpr index_t WeiVectorCount = FilterXPack * FilterH;

    // Maximum vectors per thread for global→LDS transfer
    static constexpr index_t VecsPerRow     = integer_divide_ceil(LdsStride, InVectorSize);
    static constexpr index_t MaxVecsPerThread =
        integer_divide_ceil(LdsTileH * VecsPerRow, BlockSize);

    // Horizontal padding vector type (matching original CK: vector_type<InDataType, Pad_W>)
    using HorizontalPaddingVector = ext_vector_t<InDataType, PadW>;

    /**
     * @brief Main pipeline operator.
     */
    CK_TILE_DEVICE void operator()(const InDataType* p_in_base,
                                   const WeiDataType* p_wei_base,
                                   OutDataType* p_out_base,
                                   char* smem,
                                   index_t Hi,
                                   index_t Wi,
                                   index_t Ho,
                                   index_t Wo,
                                   index_t in_h_stride,
                                   index_t in_w_stride,
                                   index_t in_n_stride,
                                   index_t wei_y_stride,
                                   index_t wei_x_stride,
                                   index_t out_h_stride,
                                   index_t out_w_stride,
                                   index_t out_n_stride) const
    {
        const index_t lane_id = __lane_id();

        // Calculate number of tiles in spatial dimensions
        const index_t num_h_tiles    = integer_divide_ceil(Ho, TileOutH);
        const index_t num_w_tiles    = integer_divide_ceil(Wo, TileOutW);
        const index_t tiles_per_batch = num_h_tiles * num_w_tiles;

        // Number of batch groups processed by this block
        constexpr index_t num_batch_groups = NBatch / TilePerWave;
        const index_t num_loop = num_batch_groups * tiles_per_batch;

        // LDS buffer pointer
        InDataType* lds_in = reinterpret_cast<InDataType*>(smem);

        // Temporary buffer for global→LDS transfer
        // IMPORTANT: Size must match original CK's TMP_IN_SIZE, NOT MaxVecsPerThread!
        // Original CK: constexpr index_t TMP_IN_SIZE = (LDS_TileH * Tile_In_Stride + InScalarPerVector - 1) / InScalarPerVector;
        // This is needed because we access tmp_in[i * BlockSize] in load/write functions
        constexpr index_t TmpInSize = (LdsTileH * LdsStride + InVectorSize - 1) / InVectorSize;
        InVector tmp_in[TmpInSize];

        // Load filter weights to registers
        WeiVector weight[WeiVectorCount]     = {};
        WeiVector weight_odd[WeiVectorCount] = {};
        LoadFilterWeights(p_wei_base, wei_y_stride, wei_x_stride, weight, weight_odd);

        // Calculate thread's position within tile
        const index_t lane_in_tile = lane_id % ThreadPerTile;
        const index_t tile_idx     = lane_id / ThreadPerTile;
        const index_t x_repeat     = lane_in_tile % WRepeats;
        const index_t y_repeat     = lane_in_tile / WRepeats;

        // SubTile offsets
        const index_t y_subtile = y_repeat * SubTileH;
        const index_t x_subtile = x_repeat * SubTileW;

        // LDS offset for this thread's subtile
        const index_t subtile_lds_offset =
            tile_idx * LdsTileSize + y_subtile * StrideH * LdsStride + x_subtile * StrideW;

        // Output pointer offset for this thread
        const long_index_t out_tile_offset =
            static_cast<long_index_t>(tile_idx) * out_n_stride +
            static_cast<long_index_t>(y_subtile) * out_h_stride +
            static_cast<long_index_t>(x_subtile) * out_w_stride;

        // Thread position within tile (for debug output)
        [[maybe_unused]] const index_t in_x = lane_id % (integer_divide_ceil(TileInW, InVectorSize));
        [[maybe_unused]] const index_t in_y_offset = lane_id / (integer_divide_ceil(TileInW, InVectorSize));

#ifdef CK_TILE_DEPTHWISE_DEBUG
        // Print kernel instance info (matching original CK format exactly - single printf)
        if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        {
            printf("======================= KERNEL INSTANCE INFO =======================\n"
                "TEMPLATE: BlockSize=%d, NBatch=%d, SubTile[%dx%d], TileSize[%dx%d]\n"
                "  Filter: %dx%d, Stride[%dx%d], Pad[%dx%d], Dilation[1x1]\n"
                "\n"
                "DIMENSIONS:\n"
                "  Input: [%dx%d] (actual) -> Tile[%dx%d] -> LDS[%dx%d] (with padding)\n"
                "  Work: %d H_tiles × %d W_tiles = %d iterations, %d batch_groups\n"
                "\n"
                "ORGANIZATION: %d SubTile[%dx%d], TilePerWave=%d, ThreadPerTile=%d\n"
                "VECTORIZATION: InVec=%d, OutVec=%d, WeiVec=%d\n"
                "LDS: %zu bytes (%.1f%% usage), Stride=%d\n"
                "\n"
                "DATA MOVEMENT VERIFICATION:\n"
                "  Actual Input[%dx%d] -> LDS data region[%d:%d,%d:%d], Padding=%d\n"
                "====================================================================\n",
                BlockSize, NBatch, SubTileH, SubTileW, TileInH, TileInW,
                FilterH, FilterW, StrideH, StrideW, PadH, PadW,
                Hi, Wi, TileInH, TileInW, LdsTileH, LdsTileW,
                num_h_tiles, num_w_tiles, num_loop, num_batch_groups,
                HRepeats * WRepeats, SubTileH, SubTileW, TilePerWave, ThreadPerTile,
                InVectorSize, OutVectorSize, WeiVectorSize,
                LdsTileSize * TilePerWave * sizeof(InDataType),
                float(LdsTileSize * TilePerWave * sizeof(InDataType)) / 65536.0f * 100.0f,
                LdsStride,
                Hi, Wi, PadH, PadH + Hi - 1, PadW, PadW + Wi - 1, PadH);
        }
#endif

        // Main loop over batches and spatial tiles
        for(index_t iter = 0; iter < num_loop; ++iter)
        {
            // Decode iteration index
            const index_t batch_idx      = iter / tiles_per_batch;
            const index_t tile_idx_flat  = iter % tiles_per_batch;
            const index_t h_tile_idx     = tile_idx_flat / num_w_tiles;
            const index_t w_tile_idx     = tile_idx_flat % num_w_tiles;

            // Calculate output tile origin
            const index_t h_out_offset = h_tile_idx * TileOutH;
            const index_t w_out_offset = w_tile_idx * TileOutW;

            // Calculate input region to load (with padding consideration)
            const index_t h_in_start_ideal = h_out_offset * StrideH - PadH;
            const index_t w_in_start_ideal = w_out_offset * StrideW - PadW;

            const index_t global_h_start = max(index_t(0), h_in_start_ideal);
            const index_t global_w_start = max(index_t(0), w_in_start_ideal);
            const index_t global_h_end   = min(Hi, h_in_start_ideal + LdsTileH);
            const index_t global_w_end   = min(Wi, w_in_start_ideal + LdsTileW);

            const index_t read_h = global_h_end - global_h_start;
            const index_t read_w = global_w_end - global_w_start;

            const index_t lds_h_start = global_h_start - h_in_start_ideal;
            const index_t lds_w_start = global_w_start - w_in_start_ideal;

            // Load input tile to LDS for each batch in TilePerWave
            // Two loading strategies matching original CK:
            // 1. TilePerWave != 1: Direct Global → LDS with padding (load_global_to_lds_with_padding)
            // 2. TilePerWave == 1: Global → VGPR → LDS (load_data_from_global + write_data_to_lds)
            const index_t lds_offset_base = lds_h_start * LdsStride + lds_w_start;
            
            static_for<0, TilePerWave, 1>{}([&](auto tile_in_wave) {
                const long_index_t batch_offset =
                    static_cast<long_index_t>(batch_idx * TilePerWave + tile_in_wave) * in_n_stride;
                const auto* p_in_current = p_in_base + batch_offset +
                                           global_h_start * in_h_stride;

                const index_t tile_lds_base = tile_in_wave * LdsTileSize;
                InDataType* p_lds_tile      = lds_in + tile_lds_base;

                if constexpr(TilePerWave != 1)
                {
                    // Strategy 1: Direct Global → LDS with padding
                    // Used when processing multiple tiles per wave
                    load_global_to_lds_with_padding(p_in_current,
                                                    p_lds_tile,
                                                    read_h,
                                                    read_w,
                                                    in_h_stride,
                                                    lds_h_start,
                                                    lds_w_start);
                }
                else
                {
                    // Strategy 2: Global → VGPR → LDS (two-step)
                    // Used when processing single tile per wave
                    InDataType* p_lds_write = p_lds_tile + lds_offset_base;

#ifdef CK_TILE_DEPTHWISE_DEBUG
                    if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && tile_in_wave == 0) {
                        printf("\nTile[%d,%d] Output[%d,%d], Tile_Per_Wave=%d\n"
                               "  Output space: [%d:%d, %d:%d] (%d×%d)\n"
                               "  Input required (ideal): [%d:%d, %d:%d]\n"
                               "  Global read (actual): [%d:%d, %d:%d] (%d×%d)\n"
                               "  LDS write: offset=[%d,%d] size=[%d×%d] LDS_size=[%d×%d]\n"
                               "  Tile LDS size: %d, Single Ping LDS size: %d\n"
                               "\n  [BeforeLoad] tile_idx=%d, read[%dx%d], global_start[%d,%d], lds_start[%d,%d]\n"
                               "               hi_stride=%d, wi_stride=%d, p_in_current offset=%ld\n",
                               h_tile_idx, w_tile_idx, h_out_offset, w_out_offset, TilePerWave,
                               h_out_offset, h_out_offset + TileOutH, w_out_offset, w_out_offset + TileOutW, TileOutH, TileOutW,
                               h_in_start_ideal, h_in_start_ideal + LdsTileH, w_in_start_ideal, w_in_start_ideal + LdsTileW,
                               global_h_start, global_h_end, global_w_start, global_w_end, read_h, read_w,
                               lds_h_start, lds_w_start, read_h, read_w, LdsTileH, LdsStride,
                               LdsTileSize, LdsTileSize * TilePerWave,
                               static_cast<int>(tile_in_wave), read_h, read_w, global_h_start, global_w_start, lds_h_start, lds_w_start,
                               in_h_stride, in_w_stride, static_cast<long>(p_in_current - p_in_base));
                    }
#endif
                    
                    load_data_from_global(p_in_current,
                                          read_h,
                                          read_w,
                                          in_h_stride,
                                          in_w_stride,
                                          tmp_in,
                                          global_w_start);
                    
                    write_data_to_lds(p_lds_write,
                                      read_h,
                                      read_w,
                                      tmp_in);
                    
                    // Clear LDS boundary padding
                    clear_lds_boundary_padding(p_lds_tile,
                                               read_h,
                                               read_w,
                                               lds_h_start,
                                               lds_w_start);
                }
            });

            block_sync_lds();

#ifdef CK_TILE_DEPTHWISE_DEBUG
            // Print iteration info and LDS dump (matching original CK format exactly - single printf)
            if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
            {
                const index_t actual_batch_start = batch_idx * TilePerWave;
                const index_t actual_batch_end = actual_batch_start + TilePerWave - 1;
                const index_t dump_num = 4;
                const bool should_dump = (iter < dump_num) || (iter >= num_loop - dump_num);
                
                if(should_dump) {
                    printf("\n===== [SERIAL] LDS DUMP: Block[%u,%u] Thread[%u,%u] Lane[%d] =====\n"
                           "Iteration %d/%d: BatchGroup %d (batches %d-%d), Tile[%d,%d]\n"
                           "Work Split: %d H_tiles × %d W_tiles = %d tiles_per_batch\n"
                           "Output[%d,%d], Input read[%d:%d,%d:%d] -> LDS[%d:%d,%d:%d]\n"
                           "Thread mapping: in[x=%d,y=%d], out[x=%d,y=%d]\n"
                           "Read from: Ping buffer (addr=%p)\n",
                           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, lane_id,
                           iter + 1, num_loop, batch_idx + 1, actual_batch_start, actual_batch_end, h_tile_idx, w_tile_idx,
                           num_h_tiles, num_w_tiles, tiles_per_batch,
                           h_out_offset, w_out_offset, global_h_start, global_h_end, global_w_start, global_w_end,
                           lds_h_start, lds_h_start + read_h, lds_w_start, lds_w_start + read_w,
                           in_x, in_y_offset, x_repeat, y_repeat,
                           static_cast<const void*>(lds_in));

                    // Dump LDS contents (matching original CK's dump_lds call)
                    dump_lds(lds_in, LdsTileSize, LdsStride);
                }
            }
#endif

            // Compute convolution for this tile
            const index_t actual_out_h = min(TileOutH, Ho - h_out_offset);
            const index_t actual_out_w = min(TileOutW, Wo - w_out_offset);
            const index_t effective_h  = max(index_t(0), min(SubTileH, actual_out_h - y_subtile));
            const index_t effective_w  = max(index_t(0), min(SubTileW, actual_out_w - x_subtile));

            // Calculate output pointer for this iteration
            const long_index_t batch_out_offset =
                static_cast<long_index_t>(batch_idx * TilePerWave) * out_n_stride;
            const long_index_t spatial_out_offset =
                static_cast<long_index_t>(h_out_offset) * out_h_stride +
                static_cast<long_index_t>(w_out_offset) * out_w_stride;
            auto* p_out_current = p_out_base + batch_out_offset + spatial_out_offset + out_tile_offset;

            // Run convolution computation
            InVectorInternal* p_lds_subtile =
                reinterpret_cast<InVectorInternal*>(lds_in + subtile_lds_offset);

            RunConvolution(p_lds_subtile,
                           weight,
                           weight_odd,
                           p_out_current,
                           out_h_stride,
                           out_w_stride,
                           effective_h,
                           effective_w);
        }
    }

private:
    /**
     * @brief Load filter weights to registers.
     */
    CK_TILE_DEVICE static void LoadFilterWeights(const WeiDataType* p_wei,
                                                  index_t wei_y_stride,
                                                  index_t wei_x_stride,
                                                  WeiVector* weight,
                                                  WeiVector* weight_odd)
    {
        // Exactly matching original CK's load_filter_data
        constexpr index_t stride = integer_divide_ceil(FilterW, WeiVectorSize);
        static_for<0, FilterH, 1>{}([&](auto y) {
            static_for<0, FilterW, 1>{}([&](auto x) {
                auto* p_wei_elem = p_wei + y * wei_y_stride + x * wei_x_stride;
                weight[y * stride + x / WeiVectorSize][x % WeiVectorSize] = *p_wei_elem;
                weight_odd[y * stride + (x + 1) / WeiVectorSize][(x + 1) % WeiVectorSize] = *p_wei_elem;
            });
        });
    }

    // ==================== Strategy 1: Direct Global → LDS ====================
    /**
     * @brief Load input tile from global memory directly to LDS with padding.
     *
     * This matches original CK's load_global_to_lds_with_padding exactly.
     * Used when TilePerWave != 1.
     */
    CK_TILE_DEVICE void load_global_to_lds_with_padding(const InDataType* p_global,
                                                         InDataType* p_lds,
                                                         index_t src_h,
                                                         index_t src_w,
                                                         index_t global_h_stride,
                                                         index_t pad_top,
                                                         index_t pad_left) const
    {
        const index_t tid = threadIdx.x;

        // Stage 1: Zero entire LDS tile
        constexpr index_t total_lds_vecs = LdsTileH * (LdsStride / InVectorSize);
        constexpr index_t clear_iters    = integer_divide_ceil(total_lds_vecs, BlockSize);

        InVector zero_vec{};
        __builtin_memset(&zero_vec, 0, sizeof(zero_vec));

        auto* p_lds_vector = reinterpret_cast<InVector*>(p_lds);

        static_for<0, clear_iters, 1>{}([&](auto iter) {
            const index_t vec_idx = tid + iter * BlockSize;
            if(vec_idx < total_lds_vecs)
            {
                p_lds_vector[vec_idx] = zero_vec;
            }
        });

        block_sync_lds();

        // Stage 2: Load data using row grouping strategy (matching original CK)
        constexpr index_t aligned_pack_w = integer_divide_ceil(LdsTileW, InVectorSize);
        const index_t num_groups         = BlockSize / aligned_pack_w;
        const index_t pack_h             = src_h / num_groups;
        const index_t remainder_rows     = src_h % num_groups;

        // Calculate 2D coordinates from thread ID
        const index_t x         = tid % aligned_pack_w;
        const index_t y_offset  = tid / aligned_pack_w;
        const index_t x_offset  = x * InVectorSize;

        const index_t vectors_per_row    = src_w / InVectorSize;
        const index_t remaining_scalars  = src_w % InVectorSize;
        const bool has_boundary          = remaining_scalars > 0;

        // Load main rows
        for(index_t group_idx = 0; group_idx < pack_h; ++group_idx)
        {
            const index_t row_y = y_offset + group_idx * num_groups;
            const InDataType* global_addr = p_global + row_y * global_h_stride + x_offset;

            const index_t lds_scalar_offset = (pad_top + row_y) * LdsStride + pad_left + x_offset;
            InDataType* lds_scalar_addr = p_lds + lds_scalar_offset;

            if(x < vectors_per_row)
            {
                InVector tmp_vec;
                __builtin_memcpy(&tmp_vec, global_addr, sizeof(InVector));
                __builtin_memcpy(lds_scalar_addr, &tmp_vec, sizeof(InVector));
            }
            else if(has_boundary && x == vectors_per_row)
            {
                // Use static_for for boundary scalars (matching original CK)
                static_for<0, InVectorSize, 1>{}([&](auto i) {
                    if(i < remaining_scalars)
                    {
                        lds_scalar_addr[i] = global_addr[i];
                    }
                });
            }
        }

        // Load remaining rows
        if(remainder_rows > 0 && y_offset < remainder_rows)
        {
            const index_t row_y = y_offset + pack_h * num_groups;
            const InDataType* global_addr = p_global + row_y * global_h_stride + x_offset;

            const index_t lds_scalar_offset = (pad_top + row_y) * LdsStride + pad_left + x_offset;
            InDataType* lds_scalar_addr = p_lds + lds_scalar_offset;

            if(x < vectors_per_row)
            {
                InVector tmp_vec;
                __builtin_memcpy(&tmp_vec, global_addr, sizeof(InVector));
                __builtin_memcpy(lds_scalar_addr, &tmp_vec, sizeof(InVector));
            }
            else if(has_boundary && x == vectors_per_row)
            {
                // Use static_for for boundary scalars (matching original CK)
                static_for<0, InVectorSize, 1>{}([&](auto i) {
                    if(i < remaining_scalars)
                    {
                        lds_scalar_addr[i] = global_addr[i];
                    }
                });
            }
        }
    }

    // ==================== Strategy 2: Global → VGPR → LDS ====================
    // Constants for padding clearing (matching original CK exactly)
    // Original CK line 147: static constexpr index_t VerticalPaddingVecs = Pad_H * VecsPerRow;
    // Original CK line 148: static constexpr index_t VerticalPaddingIters = math::integer_divide_ceil(VerticalPaddingVecs, BlockSize);
    // Original CK line 149: static constexpr index_t HorizontalPaddingIters = math::integer_divide_ceil(LDS_TileH, BlockSize);
    static constexpr index_t VerticalPaddingVecs = PadH * VecsPerRow;
    static constexpr index_t VerticalPaddingIters = integer_divide_ceil(VerticalPaddingVecs, BlockSize);
    static constexpr index_t HorizontalPaddingIters = integer_divide_ceil(LdsTileH, BlockSize);

    /**
     * @brief Load data from global memory to VGPR.
     *
     * This is copied exactly from original CK's load_data_from_global.
     * Used when TilePerWave == 1.
     */
    CK_TILE_DEVICE void load_data_from_global(const InDataType* p_global,
                                               index_t src_h,
                                               index_t src_w,
                                               index_t global_h_stride,
                                               index_t global_w_stride,
                                               InVector* tmp_in,
                                               index_t col_offset) const
    {
        const index_t tid = threadIdx.x + threadIdx.y * blockDim.x;
        const index_t total_threads = blockDim.x * blockDim.y;

        // Use ck_tile's tensor descriptor and buffer view
        auto src_desc = make_naive_tensor_descriptor(
            make_tuple(src_h, src_w),
            make_tuple(global_h_stride, global_w_stride));
        
        const index_t src_virtual_size = src_h * integer_least_multiple(global_h_stride, InVectorSize);
        auto src_buf = make_buffer_view<address_space_enum::global>(
            const_cast<InDataType*>(p_global), src_virtual_size);
        
        using src_vector_t = ext_vector_t<InDataType, InVectorSize>;
        const index_t vecs_per_row = (src_w + InVectorSize - 1) / InVectorSize;
        const index_t total_vecs = src_h * vecs_per_row;
        
        // Cooperatively load: Global Memory -> VGPR (vector load only)
        static_for<0, MaxVecsPerThread, 1>{}([&](auto i) {
            const index_t vec_idx = tid + i * total_threads;
            const index_t row = vec_idx / vecs_per_row;
            const index_t vec_in_row = vec_idx - row * vecs_per_row;
            const index_t base_col = vec_in_row * InVectorSize;
            
            const index_t global_col = col_offset + base_col;
            auto coord = make_tensor_coordinate(src_desc, make_multi_index(row, global_col));
            const bool is_valid = coordinate_has_valid_offset_assuming_top_index_is_valid(src_desc, coord);
            const bool is_last_vec = (vec_idx == total_vecs - 1);
            const bool need_shift = is_last_vec && (src_w % InVectorSize != 0);

            const index_t src_offset = coord.get_offset() - (__builtin_expect(need_shift, false) ? PadW : 0);
            auto loaded_buf = src_buf.template get<src_vector_t>(src_offset, 0, is_valid);
            src_vector_t loaded_vec = bit_cast<src_vector_t>(loaded_buf);
            
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-reinterpret-cast"
            tmp_in[i * BlockSize] = __builtin_expect(need_shift, false)
                ? *reinterpret_cast<const src_vector_t*>(reinterpret_cast<const InDataType*>(&loaded_vec) + PadW)
                : loaded_vec;
#pragma clang diagnostic pop
        });
    }

    /**
     * @brief Write data from VGPR to LDS.
     *
     * This matches original CK's write_data_to_lds exactly.
     * Used when TilePerWave == 1.
     */
    CK_TILE_DEVICE void write_data_to_lds(InDataType* p_lds,
                                           index_t src_h,
                                           index_t src_w,
                                           const InVector* tmp_in) const
    {
        const index_t tid = threadIdx.x + threadIdx.y * blockDim.x;
        const index_t total_threads = blockDim.x * blockDim.y;

        const index_t vecs_per_row = (src_w + InVectorSize - 1) / InVectorSize;
        [[maybe_unused]] const index_t total_vecs = src_h * vecs_per_row;

        // Cooperatively write: VGPR -> LDS (vector write)
        auto* p_lds_vec = reinterpret_cast<InVector*>(p_lds);

        static_for<0, MaxVecsPerThread, 1>{}([&](auto i) {
            const index_t vec_idx = tid + i * total_threads;
            const index_t row = vec_idx / vecs_per_row;
            const index_t vec_in_row = vec_idx - row * vecs_per_row;
            const index_t base_col = vec_in_row * InVectorSize;

            const index_t lds_vec_idx = (row * LdsStride + base_col) / InVectorSize;

            p_lds_vec[lds_vec_idx] = tmp_in[i * BlockSize];
        });
    }

    /**
     * @brief Clear LDS boundary padding regions.
     *
     * This matches original CK's clear_lds_boundary_padding exactly.
     * Uses vector writes for efficiency, no runtime for loops.
     * 
     * Note: Original CK uses lane_id (0-63) which equals threadIdx.x when BlockSize=64.
     */
    CK_TILE_DEVICE void clear_lds_boundary_padding(InDataType* p_lds,
                                                    index_t data_height,
                                                    index_t data_width,
                                                    index_t pad_top,
                                                    index_t pad_left) const
    {
        // Use lane_id to match original CK exactly (lane_id == threadIdx.x when BlockSize=64)
        const index_t lane_id = __lane_id();

        // Vertical zero vector for clearing rows (using InVector, matching original CK's InDataVector)
        InVector vertical_zero_vec{};
        __builtin_memset(&vertical_zero_vec, 0, sizeof(vertical_zero_vec));

        // Horizontal zero vector for clearing columns 
        // Matching original CK: using HorizontalPaddingVector = typename vector_type<InDataType, Pad_W>::type;
        HorizontalPaddingVector horizontal_zero_vec{};
        __builtin_memset(&horizontal_zero_vec, 0, sizeof(horizontal_zero_vec));

        const index_t data_end_row = pad_top + data_height;
        const index_t bottom_rows = LdsTileH - data_end_row;

        // Clear top padding rows (matching original CK exactly)
        if(pad_top > 0)
        {
            static_for<0, VerticalPaddingIters, 1>{}([&](auto iter) {
                const index_t vec_idx = lane_id + iter * BlockSize;
                if(vec_idx < VerticalPaddingVecs)
                {
                    InDataType* ptr = p_lds + vec_idx * InVectorSize;
                    *reinterpret_cast<InVector*>(__builtin_assume_aligned(ptr, alignof(InVector))) = vertical_zero_vec;
                }
            });
        }

        // Clear bottom padding rows (matching original CK exactly)
        if(bottom_rows > 0)
        {
            InDataType* bottom_base = p_lds + data_end_row * LdsStride;
            static_for<0, VerticalPaddingIters, 1>{}([&](auto iter) {
                const index_t vec_idx = lane_id + iter * BlockSize;
                if(vec_idx < VerticalPaddingVecs)
                {
                    InDataType* ptr = bottom_base + vec_idx * InVectorSize;
                    *reinterpret_cast<InVector*>(__builtin_assume_aligned(ptr, alignof(InVector))) = vertical_zero_vec;
                }
            });
        }

        // Clear left padding columns (matching original CK exactly)
        if(pad_left > 0)
        {
            static_for<0, HorizontalPaddingIters, 1>{}([&](auto iter) {
                const index_t row = lane_id + iter * BlockSize;
                if(row < LdsTileH)
                {
                    InDataType* row_base = p_lds + row * LdsStride;
                    *reinterpret_cast<HorizontalPaddingVector*>(__builtin_assume_aligned(row_base, alignof(HorizontalPaddingVector))) = horizontal_zero_vec;
                }
            });
        }

        // Clear right padding columns (matching original CK exactly)
        const index_t pad_right = LdsStride - pad_left - data_width;
        if(pad_right > 0)
        {
            static_for<0, HorizontalPaddingIters, 1>{}([&](auto iter) {
                const index_t row = lane_id + iter * BlockSize;
                if(row < LdsTileH)
                {
                    InDataType* right_base = p_lds + row * LdsStride + pad_left + data_width;
                    *reinterpret_cast<HorizontalPaddingVector*>(__builtin_assume_aligned(right_base, alignof(HorizontalPaddingVector))) = horizontal_zero_vec;
                }
            });
        }
    }

    /**
     * @brief Run convolution computation using circular buffer.
     */
    CK_TILE_DEVICE void RunConvolution(InVectorInternal* p_lds_subtile,
                                        const WeiVector* weight,
                                        const WeiVector* weight_odd,
                                        OutDataType* p_out,
                                        index_t out_h_stride,
                                        index_t out_w_stride,
                                        index_t h_max,
                                        index_t w_max) const
    {
        using InData2 = ext_vector_t<InDataType, 2>;

        // Calculate input data dimensions for subtile
        constexpr index_t SubTileInW =
            integer_least_multiple(SubTileW * StrideW + (FilterW - 1), InVectorSizeInternal);

        // Circular buffer for FilterH rows of input data
        InVectorInternal tmp_in[FilterH][SubTileInW / InVectorSizeInternal];

        // Lambda to read one row from LDS
        auto get_in = [&](index_t hi, auto count, auto* input) {
            static_for<0, count / InVectorSizeInternal, 1>{}([&](auto wi) {
                input[wi] = p_lds_subtile[hi * LdsStride / InVectorSizeInternal + wi];
            });
        };

        // Lambda to write output
        // IMPORTANT: Use OutVectorSizeInternal (not OutVectorSize) to match original CK's OutScalarPerVector_Internal
        constexpr index_t OutVecInternal = Traits::OutVectorSizeInternal;
        auto set_out = [&](index_t ho, auto count, AccDataType* acc) {
            static_for<0, count / OutVecInternal, 1>{}([&](auto wo) {
                typename Traits::OutVectorInternal output = {};
                static_for<0, OutVecInternal, 1>{}([&](auto i) {
                    output[i.value] = type_convert<OutDataType>(acc[wo * OutVecInternal + i]);
                });

                if(ho < h_max && wo * OutVecInternal < w_max)
                {
                    OutDataType* row_ptr   = p_out + ho * out_h_stride;
                    const index_t col_offset = wo * OutVecInternal * out_w_stride;
                    const index_t remaining  = w_max - col_offset;

                    if(remaining >= OutVecInternal)
                    {
                        __builtin_memcpy(row_ptr + col_offset, &output,
                                         sizeof(typename Traits::OutVectorInternal));
                    }
                    else
                    {
                        for(index_t i = 0; i < remaining; ++i)
                        {
                            row_ptr[col_offset + i] = output[i];
                        }
                    }
                }
            });
        };

        // Preload first (FilterH - StrideH) rows
        static_for<0, FilterH - StrideH, 1>{}(
            [&](auto hi) { get_in(hi, number<SubTileInW>{}, tmp_in[hi]); });

        // Main loop over SubTileH output rows
        static_for<0, SubTileH, 1>{}([&](auto ho) {
            AccDataType tmp_out[SubTileW] = {};

            // Update circular buffer: load new StrideH rows
            static_for<0, StrideH, 1>{}([&](auto s) {
                constexpr index_t hi        = ho * StrideH + FilterH - StrideH + s;
                constexpr index_t tmp_y_idx = (ho * StrideH + FilterH - StrideH + s) % FilterH;
                get_in(hi, number<SubTileInW>{}, tmp_in[tmp_y_idx]);
            });

            constexpr index_t wo_step = (StrideW == 1 && SubTileW >= 2) ? 2 : 1;

            // Iterate over SubTileW output columns
            static_for<0, SubTileW, wo_step>{}([&](auto wo) {
                // Iterate over filter
                static_for<0, FilterH, 1>{}([&](auto y) {
                    static_for<0, FilterXPack, 1>{}([&](auto x_pack) {
                        const InData2* p_in =
                            reinterpret_cast<const InData2*>(tmp_in[(ho * StrideH + y) % FilterH]) +
                            wo * StrideW / 2 + x_pack;

                        depthwise_inner_product(*p_in, weight[y * FilterXPack + x_pack], tmp_out[wo.value]);

                        if constexpr(StrideW == 1 && wo_step == 2 && wo.value < SubTileW - 1)
                        {
                            depthwise_inner_product(
                                *p_in, weight_odd[y * FilterXPack + x_pack], tmp_out[wo.value + 1]);
                        }
                    });
                });
            });

            set_out(ho, number<SubTileW>{}, tmp_out);
        });
    }
};

} // namespace ck_tile

