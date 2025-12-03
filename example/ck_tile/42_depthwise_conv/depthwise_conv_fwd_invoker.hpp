// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "depthwise_conv_utils.hpp"
#include "ck_tile/ops/depthwise_conv.hpp"
#include <limits>
#include <vector>
#include <functional>
#include <tuple>
#include <cmath>
#include <iomanip>

// Debug flag - uncomment to enable global output dump
// #define CK_TILE_DEPTHWISE_DEBUG_OUTPUT_DUMP

namespace ck_tile {

/**
 * @brief Verification info passed to invoker.
 */
template <typename OutDataType>
struct VerificationInfo
{
    bool do_verification = false;
    DeviceMem* p_out_dev = nullptr;
    HostTensor<OutDataType>* p_out_host = nullptr;
    const OutDataType* p_out_ref = nullptr;
    std::size_t output_size = 0;
};

/**
 * @brief Result of running a single kernel configuration.
 */
struct KernelRunResult
{
    float time_ms;
    bool is_valid;
    bool is_verified;
    std::string config_name;
    float tflops;
    float gb_per_sec;
};

/**
 * @brief Verify GPU result against CPU reference with detailed error reporting.
 * 
 * Matches CK format:
 *   out[8] != ref[8]: -4 != -6
 *   max err: 215, number of errors: 97940706, 84.71976% wrong values
 */
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
    constexpr int max_print_errors = 4;  // Match CK: print first 4 errors

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
            // Print first few errors in CK format
            if(print_errors && printed_errors < max_print_errors)
            {
                std::cout << "\tout[" << i << "] != ref[" << i << "]: " 
                          << static_cast<int>(p_gpu[i]) << " != " << static_cast<int>(p_cpu[i]) << std::endl;
                printed_errors++;
            }
        }
    }

    if(error_count > 0 && print_errors)
    {
        double error_pct = 100.0 * static_cast<double>(error_count) / static_cast<double>(size);
        std::cout << "max err: " << static_cast<int>(max_err) 
                  << ", number of errors: " << error_count 
                  << ", " << std::fixed << std::setprecision(5) << error_pct << "% wrong values" << std::endl;
    }

    return error_count == 0;
}

/**
 * @brief Invoker for depthwise convolution forward pass.
 *
 * This invoker follows the same pattern as CK:
 * 1. Define a set of fixed instances (configurations) matching original CK
 * 2. At runtime, iterate through all instances
 * 3. Skip instances that don't meet the requirements (IsSupportedArgument)
 * 4. Run all matching instances
 * 5. Select the best performing one
 */
struct DepthwiseConvFwdInvoker
{
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType>
    static std::tuple<float, std::string, bool, int> depthwise_conv_fwd(
        const DepthwiseConvFwdHostArgs& args,
        const stream_config& s,
        const VerificationInfo<OutDataType>& verify_info,
        std::size_t flop = 0,
        std::size_t num_byte = 0)
    {
        // Run all instances and find the best one
        return run_all_instances<InDataType, WeiDataType, AccDataType, OutDataType>(
            args, s, verify_info, flop, num_byte);
    }

private:
    /**
     * @brief Try a specific kernel configuration.
     *
     * Template parameters match original CK DeviceGroupedConvFwdDlV5:
     * - TileH, TileW: Output tile size
     * - FilterSize: Filter size (3, 5, 7, 9)
     * - DilationH, DilationW: Dilation
     * - StrideH, StrideW: Stride
     * - PadH, PadW: Padding
     * - NBatch: Number of batches per block
     * - SubTileH, SubTileW: Sub-tile size
     * - InVec, OutVec: Vectorization widths
     */
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              index_t TileH,
              index_t TileW,
              index_t FilterSize,
              index_t DilationH,
              index_t DilationW,
              index_t StrideH,
              index_t StrideW,
              index_t PadH,
              index_t PadW,
              index_t NBatchSize,
              index_t SubTileH,
              index_t SubTileW,
              index_t InVecSize,
              index_t OutVecSize>
    static KernelRunResult try_instance(const DepthwiseConvFwdHostArgs& args,
                                        const stream_config& s,
                                        const VerificationInfo<OutDataType>& verify_info,
                                        int instance_idx,
                                        std::size_t flop,
                                        std::size_t num_byte)
    {
        KernelRunResult result;
        result.is_valid = false;
        result.is_verified = true;  // Assume true if no verification
        result.time_ms = std::numeric_limits<float>::max();
        result.tflops = 0.0f;
        result.gb_per_sec = 0.0f;

        // Early exit: Check if this instance matches the problem's filter/stride/dilation/pad
        // Do this BEFORE building config name string to avoid unnecessary allocation
        if(args.Y != FilterSize || args.X != FilterSize)
            return result;
        if(args.stride_h != StrideH || args.stride_w != StrideW)
            return result;
        if(args.dilation_h != DilationH || args.dilation_w != DilationW)
            return result;
        if(args.pad_h != PadH || args.pad_w != PadW)
            return result;

        // Build config name matching CK format (only if we passed the checks above)
        std::ostringstream oss;
        oss << "DeviceGroupedConvFwdDlV5<2, 64, "
            << "FilterSize<" << FilterSize << "," << FilterSize << ">, "
            << "Dilation<" << DilationH << ", " << DilationW << ">, "
            << "Stride<" << StrideH << ", " << StrideW << ">, "
            << "NBatch: " << NBatchSize << ", "
            << "SubTileH: " << SubTileH << ", SubTileW: " << SubTileW << ", "
            << "InScalarPerVector: " << InVecSize << ", OutScalarPerVector: " << OutVecSize << ", "
            << "RequirePadding: 0, "
            << "TileH: " << TileH << ", TileW: " << TileW << ">";
        result.config_name = oss.str();

        // Define traits for this configuration
        using Traits = DepthwiseConvFwdTraits<InDataType,
                                              WeiDataType,
                                              AccDataType,
                                              OutDataType,
                                              64,          // BlockSize
                                              TileH,       // TileH
                                              TileW,       // TileW
                                              FilterSize,  // FilterH
                                              FilterSize,  // FilterW
                                              StrideH,     // StrideH
                                              StrideW,     // StrideW
                                              DilationH,   // DilationH
                                              DilationW,   // DilationW
                                              PadH,        // PadH
                                              PadW,        // PadW
                                              NBatchSize,  // NBatch
                                              SubTileH,    // SubTileH
                                              SubTileW,    // SubTileW
                                              InVecSize,   // InVectorSize
                                              OutVecSize>; // OutVectorSize

        using Pipeline = DepthwiseConvFwdPipeline<Traits>;
        using Kernel   = DepthwiseConvFwdKernel<Traits, Pipeline>;

        // Check if arguments are supported
        if(!Kernel::IsSupportedArgument(args))
        {
            return result;
        }

        // Create kernel arguments
        auto kargs = Kernel::MakeKernelArgs(args);

        // Calculate grid and block dimensions
        const auto grids  = Kernel::GridSize(args.G, args.N);
        const auto blocks = Kernel::BlockSize_();

        // Launch kernel
        float time_ms = launch_kernel(s,
                                      make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

        result.is_valid = true;
        result.time_ms = time_ms;
        
        // Calculate performance metrics
        if(flop > 0 && time_ms > 0)
        {
            result.tflops = static_cast<float>(flop) / 1.E9 / time_ms;
            result.gb_per_sec = static_cast<float>(num_byte) / 1.E6 / time_ms;
        }

#ifdef CK_TILE_DEPTHWISE_DEBUG_OUTPUT_DUMP
        // Dump global output (matching original CK format exactly)
        if(verify_info.p_out_dev != nullptr && verify_info.p_out_host != nullptr)
        {
            // Synchronize stream
            hipStreamSynchronize(s.stream_id_);
            
            // Copy result back to host for dump
            verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());
            
            printf("\nKernel finished. Time: %f ms\n", time_ms);
            
            const index_t G = args.G;
            const index_t N = args.N;
            const index_t K = args.K;
            const index_t HO = args.Ho;
            const index_t WO = args.Wo;
            
            const index_t g_stride = args.out_strides[0];
            const index_t n_stride = args.out_strides[1];
            const index_t k_stride = args.out_strides[2];
            const index_t ho_stride = args.out_strides[3];
            const index_t wo_stride = args.out_strides[4];
            
            printf("\n=== Global Output Data Dump ===\n"
                   "Output shape: G=%d, N=%d, K=%d, HO=%d, WO=%d\n"
                   "Output strides: g=%d, n=%d, k=%d, ho=%d, wo=%d\n",
                   G, N, K, HO, WO, g_stride, n_stride, k_stride, ho_stride, wo_stride);
            
            const index_t dump_n = 32;  // default: 32
            const index_t dump_num = 2; // default: 2
            const index_t dump_g_num = 2; // default: 2
            const OutDataType* p_out = verify_info.p_out_host->data();
            
            for(index_t g = 0; g < G; g++) {
                const bool should_dump_group = (g < dump_g_num) || (g >= G - dump_g_num);
                if(!should_dump_group) continue;

                for(index_t n = 0; n < std::min(N, dump_n); n++) {
                    const bool should_dump_batch = (n < dump_num) || (n >= std::min(N, dump_n) - dump_num);
                    if(!should_dump_batch) continue;

                    printf("\n=== Dumping Group %d/%d, Batch %d/%d ===\n", g + 1, G, n + 1, std::min(N, dump_n));

                    for(index_t k = 0; k < K; k++) {
                        printf("\nGroup %d, Batch %d, Channel %d:\n", g, n, k);
                        for(index_t h = 0; h < HO; h++) {
                            printf("Row %d: ", h);
                            for(index_t w = 0; w < WO; w++) {
                                index_t offset = g * g_stride + n * n_stride + k * k_stride + 
                                            h * ho_stride + w * wo_stride;
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

        // Perform verification if enabled
        if(verify_info.do_verification && 
           verify_info.p_out_dev != nullptr && 
           verify_info.p_out_host != nullptr &&
           verify_info.p_out_ref != nullptr)
        {
            // Copy result back to host
            verify_info.p_out_dev->FromDevice(verify_info.p_out_host->data());

            // Verify against reference (print errors only for first failed instance)
            static bool first_error_printed = false;
            bool print_errors = !first_error_printed;
            
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

    /**
     * @brief Run all predefined instances and select the best one.
     *
     * Instance definitions exactly match original CK from run_depthwise_conv_fwd_dl_example.inc
     */
    // Returns: (best_time, best_config, is_verified, best_instance_idx)
    template <typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType>
    static std::tuple<float, std::string, bool, int> run_all_instances(
        const DepthwiseConvFwdHostArgs& args,
        const stream_config& s,
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

        // Helper lambda to process each instance result (auto-increments instance_count)
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

        // Macro to simplify instance definitions
        // Parameters: TileH, TileW, Filter, StrH, StrW, Pad, NBatch, SubH, SubW, InVec, OutVec
        // Note: Dilation is always 1x1 for depthwise conv
#define TRY_INSTANCE(TileH, TileW, Filter, StrH, StrW, Pad, NBatch, SubH, SubW, InVec, OutVec) \
        process_result(try_instance<InDataType, WeiDataType, AccDataType, OutDataType, \
                                    TileH, TileW, Filter, 1, 1, StrH, StrW, Pad, Pad, NBatch, SubH, SubW, InVec, OutVec>( \
                       args, s, verify_info, instance_count, flop, num_byte))

        // ==================== FilterSize = 3 (Pad = 1) ====================
        TRY_INSTANCE( 8,  8, 3, 1, 1, 1, 8, 2, 2, 2, 2);
        TRY_INSTANCE(16, 16, 3, 1, 1, 1, 8, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 3, 2, 2, 1, 2, 1, 4, 8, 8);
        TRY_INSTANCE(14, 28, 3, 2, 2, 1, 1, 2, 4, 8, 8);
        TRY_INSTANCE(28, 28, 3, 1, 1, 1, 1, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 3, 1, 1, 1, 1, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 3, 2, 2, 1, 2, 4, 4, 8, 8);

        // ==================== FilterSize = 5 (Pad = 2) ====================
        TRY_INSTANCE( 8,  8, 5, 1, 1, 2, 1, 1, 1, 1, 1);
        TRY_INSTANCE( 8,  8, 5, 1, 1, 2, 8, 2, 2, 2, 2);
        TRY_INSTANCE( 8,  8, 5, 2, 2, 2, 4, 2, 2, 2, 2);
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 1, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 5, 1, 1, 2, 8, 1, 4, 8, 8);
        TRY_INSTANCE(14, 28, 5, 2, 2, 2, 2, 2, 4, 8, 8);
        TRY_INSTANCE(16, 32, 5, 2, 2, 2, 4, 1, 8, 8, 8);
        TRY_INSTANCE(28, 28, 5, 1, 1, 2, 8, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 5, 1, 1, 2, 4, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 5, 2, 2, 2, 1, 4, 4, 8, 8);

        // ==================== FilterSize = 7 (Pad = 3) ====================
        TRY_INSTANCE( 8,  8, 7, 1, 1, 3, 1, 1, 1, 1, 1);
        TRY_INSTANCE( 8,  8, 7, 1, 1, 3, 8, 2, 2, 2, 2);
        TRY_INSTANCE( 8,  8, 7, 2, 2, 3, 4, 2, 2, 2, 2);
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 1, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 7, 1, 1, 3, 8, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 7, 2, 2, 3, 2, 1, 4, 8, 8);
        TRY_INSTANCE(14, 28, 7, 2, 2, 3, 2, 2, 4, 8, 8);
        TRY_INSTANCE(16, 32, 7, 2, 2, 3, 4, 1, 8, 8, 8);
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 1, 4, 4, 8, 8);
        TRY_INSTANCE(28, 28, 7, 1, 1, 3, 8, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 1, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 7, 1, 1, 3, 4, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 2, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 7, 2, 2, 3, 1, 4, 4, 8, 8);

        // ==================== FilterSize = 9 (Pad = 4) ====================
        TRY_INSTANCE( 8,  8, 9, 1, 1, 4, 1, 1, 1, 1, 1);
        TRY_INSTANCE( 8,  8, 9, 1, 1, 4, 8, 2, 2, 2, 2);
        TRY_INSTANCE( 8,  8, 9, 2, 2, 4, 4, 2, 2, 2, 2);
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 1, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 9, 1, 1, 4, 8, 1, 4, 8, 8);
        TRY_INSTANCE(16, 16, 9, 2, 2, 4, 2, 1, 4, 8, 8);
        TRY_INSTANCE(14, 28, 9, 2, 2, 4, 2, 2, 4, 8, 8);
        TRY_INSTANCE(16, 32, 9, 2, 2, 4, 4, 1, 8, 8, 8);
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 1, 4, 4, 8, 8);
        TRY_INSTANCE(28, 28, 9, 1, 1, 4, 8, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 1, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 9, 1, 1, 4, 4, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 2, 4, 4, 8, 8);
        TRY_INSTANCE(32, 32, 9, 2, 2, 4, 1, 4, 4, 8, 8);

#undef TRY_INSTANCE

        if(valid_count == 0)
        {
            std::cerr << "Error: No suitable kernel configuration found\n";
            return {-1.0f, "", false, -1};
        }

        return {best_time, best_config, best_verified, best_instance_idx};
    }
};

} // namespace ck_tile
