// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "depthwise_conv_utils.hpp"

// #define CK_TILE_DEPTHWISE_DEBUG_OUTPUT_DUMP

template <typename OutDataType>
struct VerificationInfo
{
    bool do_verification = false;
    ck_tile::DeviceMem* p_out_dev = nullptr;
    ck_tile::HostTensor<OutDataType>* p_out_host = nullptr;
    const OutDataType* p_out_ref = nullptr;
    std::size_t output_size = 0;
};

struct KernelRunResult
{
    float time_ms = 0.0f;
    bool is_valid = false;
    bool is_verified = true;
    std::string config_name;
    float tflops = 0.0f;
    float gb_per_sec = 0.0f;
};

template <typename OutDataType>
bool verify_gpu_result(const OutDataType* p_gpu,
                       const OutDataType* p_cpu,
                       std::size_t size,
                       bool print_errors = false,
                       double rtol = 1e-3,
                       double atol = 1e-3)
{
    std::size_t error_count = 0;
    double max_err = 0.0;
    int printed_errors = 0;
    constexpr int max_print_errors = 4;

    for(std::size_t i = 0; i < size; ++i)
    {
        double gpu_val = static_cast<double>(p_gpu[i]);
        double cpu_val = static_cast<double>(p_cpu[i]);
        double diff    = std::abs(gpu_val - cpu_val);
        double ref_val = std::abs(cpu_val);

        if(diff > max_err)
        {
            max_err = diff;
        }

        if(diff > atol + rtol * ref_val)
        {
            error_count++;
            if(print_errors && printed_errors < max_print_errors)
            {
                std::cout << "\tout[" << i << "] != ref[" << i << "]: "
                          << static_cast<float>(p_gpu[i]) << " != " << static_cast<float>(p_cpu[i]) << std::endl;
                printed_errors++;
            }
        }
    }

    if(error_count > 0 && print_errors)
    {
        double error_pct = 100.0 * static_cast<double>(error_count) / static_cast<double>(size);
        std::cout << "max err: " << std::setprecision(6) << max_err
                  << ", number of errors: " << error_count
                  << ", " << std::fixed << std::setprecision(5) << error_pct << "% wrong values" << std::endl;
    }

    return error_count == 0;
}

struct DepthwiseConvFwdInvoker
{
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType>
    static std::tuple<float, std::string, bool, int> depthwise_conv_fwd(
        const ck_tile::DepthwiseConvFwdHostArgs& args,
        const ck_tile::stream_config& s,
        const VerificationInfo<OutDataType>& verify_info,
        std::size_t flop = 0,
        std::size_t num_byte = 0)
    {
        return run_all_instances<InDataType, WeiDataType, AccDataType, OutDataType>(
            args, s, verify_info, flop, num_byte);
    }

private:
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              ck_tile::index_t TileH,
              ck_tile::index_t TileW,
              ck_tile::index_t FilterSize,
              ck_tile::index_t DilationH,
              ck_tile::index_t DilationW,
              ck_tile::index_t StrideH,
              ck_tile::index_t StrideW,
              ck_tile::index_t PadH,
              ck_tile::index_t PadW,
              ck_tile::index_t NBatchSize,
              ck_tile::index_t SubTileH,
              ck_tile::index_t SubTileW,
              ck_tile::index_t InVecSize,
              ck_tile::index_t OutVecSize>
    static KernelRunResult try_instance(const ck_tile::DepthwiseConvFwdHostArgs& args,
                                        const ck_tile::stream_config& s,
                                        const VerificationInfo<OutDataType>& verify_info,
                                        int instance_idx,
                                        std::size_t flop,
                                        std::size_t num_byte)
    {
        KernelRunResult result;
        result.time_ms = std::numeric_limits<float>::max();

        if(args.Y != FilterSize || args.X != FilterSize)
            return result;
        if(args.stride_h != StrideH || args.stride_w != StrideW)
            return result;
        if(args.dilation_h != DilationH || args.dilation_w != DilationW)
            return result;
        if(args.pad_h != PadH || args.pad_w != PadW)
            return result;

        std::ostringstream oss;
        oss << "DepthwiseConvFwd<64, "
            << "FilterSize<" << FilterSize << "," << FilterSize << ">, "
            << "Dilation<" << DilationH << ", " << DilationW << ">, "
            << "Stride<" << StrideH << ", " << StrideW << ">, "
            << "NBatch: " << NBatchSize << ", "
            << "SubTileH: " << SubTileH << ", SubTileW: " << SubTileW << ", "
            << "InVecSize: " << InVecSize << ", OutVecSize: " << OutVecSize << ", "
            << "RequirePadding: 0, "
            << "TileH: " << TileH << ", TileW: " << TileW << ">";
        result.config_name = oss.str();

        // TODO: BlockSize is hardcoded to 64; make it a template parameter when supporting other sizes
        using Traits = ck_tile::DepthwiseConvFwdTraits<InDataType,
                                                       WeiDataType,
                                                       AccDataType,
                                                       OutDataType,
                                                       64,
                                                       TileH,
                                                       TileW,
                                                       FilterSize,
                                                       FilterSize,
                                                       StrideH,
                                                       StrideW,
                                                       DilationH,
                                                       DilationW,
                                                       PadH,
                                                       PadW,
                                                       NBatchSize,
                                                       SubTileH,
                                                       SubTileW,
                                                       InVecSize,
                                                       OutVecSize>;

        using Pipeline = ck_tile::DepthwiseConvFwdPipeline<Traits>;
        using Kernel   = ck_tile::DepthwiseConvFwdKernel<Traits, Pipeline>;

        if(!Kernel::IsSupportedArgument(args))
        {
            return result;
        }

        auto kargs = Kernel::MakeKernelArgs(args);

        const auto grids  = Kernel::GridSize(args.G, args.N);
        const auto blocks = Kernel::BlockSize_();

        float time_ms = ck_tile::launch_kernel(s,
                                      ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

        result.is_valid = true;
        result.time_ms = time_ms;

        if(flop > 0 && time_ms > 0)
        {
            result.tflops = static_cast<float>(flop) / 1.E9 / time_ms;
            result.gb_per_sec = static_cast<float>(num_byte) / 1.E6 / time_ms;
        }

#ifdef CK_TILE_DEPTHWISE_DEBUG_OUTPUT_DUMP
        if(verify_info.p_out_dev != nullptr && verify_info.p_out_host != nullptr)
        {
            (void)hipStreamSynchronize(s.stream_id_);
            verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());

            printf("\nKernel finished. Time: %f ms\n", time_ms);

            const ck_tile::index_t G  = args.G;
            const ck_tile::index_t N  = args.N;
            const ck_tile::index_t K  = args.K;
            const ck_tile::index_t Ho = args.Ho;
            const ck_tile::index_t Wo = args.Wo;

            printf("\n=== Global Output Data Dump ===\n"
                   "Output shape: G=%d, N=%d, K=%d, Ho=%d, Wo=%d\n"
                   "Output strides: g=%d, n=%d, k=%d, ho=%d, wo=%d\n",
                   G, N, K, Ho, Wo,
                   args.out_strides[0], args.out_strides[1], args.out_strides[2],
                   args.out_strides[3], args.out_strides[4]);

            constexpr ck_tile::index_t dump_n     = 32;
            constexpr ck_tile::index_t dump_num   = 2;
            constexpr ck_tile::index_t dump_g_num = 2;
            const OutDataType* p_out = verify_info.p_out_host->data();

            for(ck_tile::index_t g = 0; g < G; g++)
            {
                const bool should_dump_group = (g < dump_g_num) || (g >= G - dump_g_num);
                if(!should_dump_group) continue;

                for(ck_tile::index_t n = 0; n < std::min(N, dump_n); n++)
                {
                    const bool should_dump_batch = (n < dump_num) || (n >= std::min(N, dump_n) - dump_num);
                    if(!should_dump_batch) continue;

                    printf("\n=== Dumping Group %d/%d, Batch %d/%d ===\n", g + 1, G, n + 1, std::min(N, dump_n));

                    for(ck_tile::index_t k = 0; k < K; k++)
                    {
                        printf("\nGroup %d, Batch %d, Channel %d:\n", g, n, k);
                        for(ck_tile::index_t h = 0; h < Ho; h++)
                        {
                            printf("Row %d: ", h);
                            for(ck_tile::index_t w = 0; w < Wo; w++)
                            {
                                ck_tile::index_t offset = g * args.out_strides[0] + n * args.out_strides[1] +
                                            k * args.out_strides[2] + h * args.out_strides[3] +
                                            w * args.out_strides[4];
                                float display_value = static_cast<float>(p_out[offset]);
                                printf("%.3f ", display_value);
                            }
                            printf("\n");
                        }
                    }
                }
            }
            printf("=== End Global Output Dump ===\n\n");
        }
#endif

        if(verify_info.do_verification &&
           verify_info.p_out_dev != nullptr &&
           verify_info.p_out_host != nullptr &&
           verify_info.p_out_ref != nullptr)
        {
            verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());
            static bool first_error_printed = false;
            bool print_errors = (s.log_level_ > 0) && !first_error_printed;

            result.is_verified = verify_gpu_result<OutDataType>(
                verify_info.p_out_host->data(),
                verify_info.p_out_ref,
                verify_info.output_size,
                print_errors);

            if(!result.is_verified && print_errors)
            {
                first_error_printed = true;
            }
        }

        if(s.log_level_ > 0)
        {
            std::cout << "[Instance " << instance_idx << "] ";
            if(verify_info.do_verification)
            {
                std::cout << (result.is_verified ? "[PASS]" : "[FAIL]") << " ";
            }
            std::cout << std::fixed << std::setprecision(6) << time_ms << " ms, "
                      << std::setprecision(4) << result.tflops << " TFlops, "
                      << std::setprecision(3) << result.gb_per_sec << " GB/s, "
                      << result.config_name << std::endl;
        }

        return result;
    }

    // Returns: (best_time, best_config, is_verified, best_instance_idx)
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType>
    static std::tuple<float, std::string, bool, int> run_all_instances(
        const ck_tile::DepthwiseConvFwdHostArgs& args,
        const ck_tile::stream_config& s,
        const VerificationInfo<OutDataType>& verify_info,
        std::size_t flop,
        std::size_t num_byte)
    {
        float best_time = std::numeric_limits<float>::max();
        std::string best_config;
        bool best_verified = true;
        int best_instance_idx = -1;
        int instance_count = 0;
        int valid_count = 0;

        if(s.log_level_ > 0)
        {
            std::cout << "\n=== Testing all instances ===" << std::endl;
        }

        auto process_result = [&](const KernelRunResult& result) {
            if(result.is_valid)
            {
                valid_count++;
                if(result.time_ms < best_time)
                {
                    best_time = result.time_ms;
                    best_config = result.config_name;
                    best_verified = result.is_verified;
                    best_instance_idx = instance_count;
                }
            }
            instance_count++;
        };

        // Parameters: TileH, TileW, Filter, StrH, StrW, Pad, NBatch, SubTileH, SubTileW, InVecSize, OutVecSize
        // Dilation hardcoded to 1x1 in current macro
#define TRY_INSTANCE(TileH, TileW, Filter, StrH, StrW, Pad, NBatch, SubH, SubW, InVec, OutVec) \
        process_result(try_instance<InDataType, WeiDataType, AccDataType, OutDataType, \
                                    TileH, TileW, Filter, 1, 1, StrH, StrW, Pad, Pad, NBatch, SubH, SubW, InVec, OutVec>( \
                       args, s, verify_info, instance_count, flop, num_byte))

        // ============================================================================
        // FilterSize = 3, Pad = 1
        // ============================================================================
        // --- 3x3 stride=1 ---
        TRY_INSTANCE( 8,  8, 3, 1, 1, 1, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 3, 1, 1, 1, 8, 1, 4, 8, 8);   // mid tile, large batch
        TRY_INSTANCE(16, 16, 3, 1, 1, 1, 1, 2, 2, 2, 2);   // tiny image fallback (H/W<=4)
        TRY_INSTANCE(28, 28, 3, 1, 1, 1, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 3, 1, 1, 1, 1, 4, 4, 8, 8);   // large tile, NBatch=1

        // --- 3x3 stride=2 ---
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 2, 1, 4, 8, 8);   // mid tile, NBatch=2
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 1, 4, 8, 8);   // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 2, 2, 8, 8);   // small output (Ho/Wo~7-14)
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 1, 2, 2, 2, 2);   // tiny image fallback (H/W<=4)
        TRY_INSTANCE(14, 28, 3, 2, 2, 1, 1, 2, 4, 8, 8);   // asymmetric tile
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 2, 4, 4, 8, 8);   // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 4, 4, 4, 4);   // large tile, reduced vec
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 1, 2, 8, 8, 8);   // large output (Ho/Wo~256-512)

        // ============================================================================
        // FilterSize = 5, Pad = 2
        // ============================================================================
        // --- 5x5 stride=1 ---
        TRY_INSTANCE( 8,  8, 5, 1, 1, 2, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE( 8,  8, 5, 1, 1, 2, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 1, 1, 4, 8, 8);   // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 8, 1, 4, 8, 8);   // mid tile, large batch
        TRY_INSTANCE(28, 28, 5, 1, 1, 2, 8, 4, 4, 8, 8);   // large tile, large batch
        TRY_INSTANCE(32, 32, 5, 1, 1, 2, 4, 4, 4, 8, 8);   // large tile, mid batch

        // --- 5x5 stride=2 ---
        TRY_INSTANCE( 8,  8, 5, 2, 2, 2, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE( 8,  8, 5, 2, 2, 2, 1, 2, 2, 2, 2);   // small tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 2, 2, 2, 1, 1, 4, 8, 8);   // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 5, 2, 2, 2, 1, 2, 2, 8, 8);   // small output (Ho/Wo~7-14)
        TRY_INSTANCE(14, 28, 5, 2, 2, 2, 2, 2, 4, 8, 8);   // asymmetric tile
        TRY_INSTANCE(16, 32, 5, 2, 2, 2, 4, 1, 8, 8, 8);   // wide tile
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 1, 4, 4, 4, 4);   // large tile, reduced vec
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 1, 2, 8, 8, 8);   // large output (Ho/Wo~256-512)

        // ============================================================================
        // FilterSize = 7, Pad = 3
        // ============================================================================
        // --- 7x7 stride=1 ---
        TRY_INSTANCE( 8,  8, 7, 1, 1, 3, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE( 8,  8, 7, 1, 1, 3, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 1, 1, 4, 8, 8);   // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 8, 1, 4, 8, 8);   // mid tile, large batch
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 8, 4, 4, 8, 8);   // large tile, large batch
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 4, 4, 4, 8, 8);   // large tile, mid batch

        // --- 7x7 stride=2 ---
        TRY_INSTANCE( 8,  8, 7, 2, 2, 3, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE(16, 16, 7, 2, 2, 3, 2, 1, 4, 8, 8);   // mid tile, NBatch=2
        TRY_INSTANCE(14, 28, 7, 2, 2, 3, 2, 2, 4, 8, 8);   // asymmetric tile
        TRY_INSTANCE(16, 32, 7, 2, 2, 3, 4, 1, 8, 8, 8);   // wide tile
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 2, 4, 4, 8, 8);   // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 1, 4, 4, 8, 8);   // large tile, NBatch=1

        // ============================================================================
        // FilterSize = 9, Pad = 4
        // ============================================================================
        // --- 9x9 stride=1 ---
        TRY_INSTANCE( 8,  8, 9, 1, 1, 4, 1, 1, 1, 1, 1);   // minimal config
        TRY_INSTANCE( 8,  8, 9, 1, 1, 4, 8, 2, 2, 2, 2);   // small tile, large batch
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 1, 1, 4, 8, 8);   // mid tile, NBatch=1
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 8, 1, 4, 8, 8);   // mid tile, large batch
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 8, 4, 4, 8, 8);   // large tile, large batch
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 1, 4, 4, 8, 8);   // large tile, NBatch=1
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 4, 4, 8, 8);   // large tile, mid batch

        // --- 9x9 stride=2 ---
        TRY_INSTANCE( 8,  8, 9, 2, 2, 4, 4, 2, 2, 2, 2);   // small tile, NBatch=4
        TRY_INSTANCE(16, 16, 9, 2, 2, 4, 2, 1, 4, 8, 8);   // mid tile, NBatch=2
        TRY_INSTANCE(14, 28, 9, 2, 2, 4, 2, 2, 4, 8, 8);   // asymmetric tile
        TRY_INSTANCE(16, 32, 9, 2, 2, 4, 4, 1, 8, 8, 8);   // wide tile
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 2, 4, 4, 8, 8);   // large tile, NBatch=2
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 1, 4, 4, 8, 8);   // large tile, NBatch=1

#undef TRY_INSTANCE

        if(valid_count == 0)
        {
            std::cerr << "Error: No suitable kernel configuration found\n";
            return {-1.0f, "", false, -1};
        }

        return {best_time, best_config, best_verified, best_instance_idx};
    }
};
