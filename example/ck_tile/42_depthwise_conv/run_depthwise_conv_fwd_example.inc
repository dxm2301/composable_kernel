// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "depthwise_conv_utils.hpp"
#include "depthwise_conv_fwd_invoker.hpp"
#include <iomanip>

/**
 * @brief CPU reference implementation for depthwise convolution forward.
 *
 * Layout matching original CK (common.hpp):
 *   Input:  GNCHW with strides [C*H*W, G*C*H*W, 1, W*C, C]
 *   Weight: GKCYX with strides [C*Y*X, G*C*Y*X, 1, X, 1]
 *   Output: GNKHW with strides [K*H*W, G*K*H*W, 1, W*K, K]
 *
 * For depthwise conv: C=K=1, so the layout simplifies to:
 *   Input:  [G][N][1][H][W] with strides [H*W, G*H*W, 1, W, 1]
 *   Weight: [G][1][1][Y][X] with strides [Y*X, G*Y*X, 1, X, 1]
 *   Output: [G][N][1][H][W] with strides [H*W, G*H*W, 1, W, 1]
 */
template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType>
void depthwise_conv_fwd_cpu_reference(
    const InDataType* p_in,
    const WeiDataType* p_wei,
    OutDataType* p_out,
    ck_tile::index_t G,
    ck_tile::index_t N,
    ck_tile::index_t Hi,
    ck_tile::index_t Wi,
    ck_tile::index_t Ho,
    ck_tile::index_t Wo,
    ck_tile::index_t Y,
    ck_tile::index_t X,
    ck_tile::index_t stride_h,
    ck_tile::index_t stride_w,
    ck_tile::index_t dilation_h,
    ck_tile::index_t dilation_w,
    ck_tile::index_t pad_h,
    ck_tile::index_t pad_w)
{
    // For depthwise conv: C=1, K=1 per group
    // Strides from GetGNCHWStrides (C=1):
    //   in_g_stride = C*H*W = H*W
    //   in_n_stride = G*C*H*W = G*H*W
    //   in_c_stride = 1 (not used since C=1)
    //   in_h_stride = W*C = W
    //   in_w_stride = C = 1
    const ck_tile::index_t in_g_stride = Hi * Wi;
    const ck_tile::index_t in_n_stride = G * Hi * Wi;
    const ck_tile::index_t in_h_stride = Wi;
    const ck_tile::index_t in_w_stride = 1;

    // Weight strides from GetGKCYXStrides (C=K=1):
    //   wei_g_stride = C*Y*X = Y*X
    //   wei_y_stride = X
    //   wei_x_stride = 1
    const ck_tile::index_t wei_g_stride = Y * X;
    const ck_tile::index_t wei_y_stride = X;
    const ck_tile::index_t wei_x_stride = 1;

    // Output strides from GetGNCHWStrides (K=1):
    //   out_g_stride = K*Ho*Wo = Ho*Wo
    //   out_n_stride = G*K*Ho*Wo = G*Ho*Wo
    //   out_h_stride = Wo*K = Wo
    //   out_w_stride = K = 1
    const ck_tile::index_t out_g_stride = Ho * Wo;
    const ck_tile::index_t out_n_stride = G * Ho * Wo;
    const ck_tile::index_t out_h_stride = Wo;
    const ck_tile::index_t out_w_stride = 1;

    for(ck_tile::index_t g = 0; g < G; ++g)
    {
        for(ck_tile::index_t n = 0; n < N; ++n)
        {
            for(ck_tile::index_t ho = 0; ho < Ho; ++ho)
            {
                for(ck_tile::index_t wo = 0; wo < Wo; ++wo)
                {
                    AccDataType acc = static_cast<AccDataType>(0);

                    for(ck_tile::index_t y = 0; y < Y; ++y)
                    {
                        for(ck_tile::index_t x = 0; x < X; ++x)
                        {
                            ck_tile::index_t hi = ho * stride_h + y * dilation_h - pad_h;
                            ck_tile::index_t wi = wo * stride_w + x * dilation_w - pad_w;

                            if(hi >= 0 && hi < Hi && wi >= 0 && wi < Wi)
                            {
                                // Input index using strides
                                ck_tile::index_t in_idx = g * in_g_stride + n * in_n_stride +
                                                          hi * in_h_stride + wi * in_w_stride;
                                // Weight index using strides
                                ck_tile::index_t wei_idx = g * wei_g_stride +
                                                           y * wei_y_stride + x * wei_x_stride;

                                acc += static_cast<AccDataType>(p_in[in_idx]) *
                                       static_cast<AccDataType>(p_wei[wei_idx]);
                            }
                        }
                    }

                    // Output index using strides
                    ck_tile::index_t out_idx = g * out_g_stride + n * out_n_stride +
                                               ho * out_h_stride + wo * out_w_stride;
                    p_out[out_idx] = static_cast<OutDataType>(acc);
                }
            }
        }
    }
}

// NOTE: verify_result was removed - use ck_tile::verify_gpu_result from depthwise_conv_fwd_invoker.hpp instead

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType>
int run_depthwise_conv_fwd_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    // Parse arguments
    const bool do_verification    = arg_parser.get_int("v");
    const int warmup              = arg_parser.get_int("warmup");
    const int repeat              = arg_parser.get_int("repeat");
    const std::string timer_type  = arg_parser.get_str("timer");
    const int init_method         = arg_parser.get_int("init");

    const ck_tile::index_t G        = arg_parser.get_int("G");
    const ck_tile::index_t N        = arg_parser.get_int("N");
    const ck_tile::index_t H        = arg_parser.get_int("H");
    const ck_tile::index_t W        = arg_parser.get_int("W");
    const ck_tile::index_t Y        = arg_parser.get_int("Y");
    const ck_tile::index_t X        = arg_parser.get_int("X");
    const ck_tile::index_t stride_h = arg_parser.get_int("stride_h");
    const ck_tile::index_t stride_w = arg_parser.get_int("stride_w");
    const ck_tile::index_t dilation_h = arg_parser.get_int("dilation_h");
    const ck_tile::index_t dilation_w = arg_parser.get_int("dilation_w");
    const ck_tile::index_t pad_h    = arg_parser.get_int("pad_h");
    const ck_tile::index_t pad_w    = arg_parser.get_int("pad_w");

    // For depthwise conv, C=1 and K=1 per group
    constexpr ck_tile::index_t C = 1;
    constexpr ck_tile::index_t K = 1;

    // Create convolution parameters
    ck_tile::DepthwiseConvParam conv_param(
        2,  // num_dim_spatial
        G,
        N,
        C,
        K,
        {H, W},           // input_spatial_lengths
        {Y, X},           // filter_spatial_lengths
        {stride_h, stride_w},
        {dilation_h, dilation_w},
        {pad_h, pad_w},   // input_left_pads
        {pad_h, pad_w}    // input_right_pads
    );

    // Calculate output dimensions
    const ck_tile::index_t Ho = conv_param.output_spatial_lengths_[0];
    const ck_tile::index_t Wo = conv_param.output_spatial_lengths_[1];

    // Get precision string
    std::string prec_str;
    if constexpr(std::is_same_v<InDataType, ck_tile::fp16_t>)
        prec_str = "fp16";
    else if constexpr(std::is_same_v<InDataType, ck_tile::bf16_t>)
        prec_str = "bf16";
    else
        prec_str = "fp32";

    // Calculate strides for GNCHW layout (matching original CK from common.hpp)
    // NOTE: These are reused for both printing and args, avoiding duplicate calculation
    const auto in_strides  = ck_tile::GetGNCHWStrides(G, N, C, H, W);
    const auto wei_strides = ck_tile::GetGKCYXStrides(G, K, C, Y, X);
    const auto out_strides = ck_tile::GetGNCHWStrides(G, N, K, Ho, Wo);

    // Print command line equivalent at the beginning (like CK format)
    std::cout << "\n=== Command ===" << std::endl;
    std::cout << "./bin/tile_example_depthwise_conv_fwd"
              << " -G=" << G << " -N=" << N << " -H=" << H << " -W=" << W
              << " -Y=" << Y << " -X=" << X
              << " -stride_h=" << stride_h << " -stride_w=" << stride_w
              << " -dilation_h=" << dilation_h << " -dilation_w=" << dilation_w
              << " -pad_h=" << pad_h << " -pad_w=" << pad_w
              << " -prec=" << prec_str
              << " -v=" << (do_verification ? 1 : 0) << std::endl;

    // Print dimension info in CK format
    std::cout << "in: dim 5, lengths {" << G << ", " << N << ", " << C << ", " << H << ", " << W << "}, "
              << "strides {" << in_strides[0] << ", " << in_strides[1] << ", " << in_strides[2] 
              << ", " << in_strides[3] << ", " << in_strides[4] << "}" << std::endl;
    std::cout << "wei: dim 5, lengths {" << G << ", " << K << ", " << C << ", " << Y << ", " << X << "}, "
              << "strides {" << wei_strides[0] << ", " << wei_strides[1] << ", " << wei_strides[2]
              << ", " << wei_strides[3] << ", " << wei_strides[4] << "}" << std::endl;
    std::cout << "out: dim 5, lengths {" << G << ", " << N << ", " << K << ", " << Ho << ", " << Wo << "}, "
              << "strides {" << out_strides[0] << ", " << out_strides[1] << ", " << out_strides[2]
              << ", " << out_strides[3] << ", " << out_strides[4] << "}" << std::endl;

    // Calculate sizes (GNCHW layout for input/output, GKCYX for weight)
    const std::size_t input_size = static_cast<std::size_t>(G) * N * C * H * W;
    const std::size_t weight_size = static_cast<std::size_t>(G) * K * C * Y * X;
    const std::size_t output_size = static_cast<std::size_t>(G) * N * K * Ho * Wo;

    // Print element sizes in CK format
    std::cout << "in elem_size: " << input_size << std::endl;
    std::cout << "wei elem_size: " << weight_size << std::endl;
    std::cout << "out elem_size: " << output_size << std::endl;

    // Allocate host memory
    ck_tile::HostTensor<InDataType> in_host({input_size});
    ck_tile::HostTensor<WeiDataType> wei_host({weight_size});
    ck_tile::HostTensor<OutDataType> out_host({output_size});
    ck_tile::HostTensor<OutDataType> out_ref({output_size});  // CPU reference

    // Initialize input data based on init_method
    // 0 = zero, 1 = integer [-5,5], 2 = random [-1,1]
    // 3 = column-wise (w % 64), weight all 1s  -- MATCHES ORIGINAL CK DEBUG MODE
    // 4 = row pattern (h), weight all 1s
    // 5 = column-wise (w % 64), weight spatial pattern (fh * FW + fw)
    switch(init_method)
    {
    case 0:
        // Zero initialization (for debugging)
        std::fill(in_host.begin(), in_host.end(), InDataType{0});
        std::fill(wei_host.begin(), wei_host.end(), WeiDataType{0});
        break;
    case 1:
        // Integer values [-5, 5] (matching CK's GeneratorTensor_2)
        ck_tile::FillUniformDistributionIntegerValue<InDataType>{-5.f, 5.f}(in_host);
        ck_tile::FillUniformDistributionIntegerValue<WeiDataType>{-5.f, 5.f}(wei_host);
        break;
    case 3:
        // Column-wise pattern (w % 64) + weight all 1s -- MATCHES ORIGINAL CK's init_input_data(true) + init_weight_data(false)
        {
            // Input: column-wise increment (matching original CK exactly)
            // Pattern: each column has value = w % 64
            auto* in_ptr = in_host.data();
            for(ck_tile::index_t g = 0; g < G; g++) {
                for(ck_tile::index_t n = 0; n < N; n++) {
                    for(ck_tile::index_t c = 0; c < C; c++) {
                        for(ck_tile::index_t h = 0; h < H; h++) {
                            for(ck_tile::index_t w = 0; w < W; w++) {
                                // Use same stride calculation as original CK
                                ck_tile::index_t offset = g * in_strides[0] + n * in_strides[1] + 
                                                          c * in_strides[2] + h * in_strides[3] + w * in_strides[4];
                                in_ptr[offset] = static_cast<InDataType>(w % 64);
                            }
                        }
                    }
                }
            }
            // Weight: all 1s (matching original CK's init_weight_data(false))
            std::fill(wei_host.begin(), wei_host.end(), WeiDataType{1});
            std::cout << "Input: column-wise (w % 64), Weight: all 1s (MATCHES ORIGINAL CK)" << std::endl;
        }
        break;
    case 4:
        // Row pattern (h) + weight all 1s -- MATCHES ORIGINAL CK's init_input_data(false) + init_weight_data(false)
        {
            auto* in_ptr = in_host.data();
            for(ck_tile::index_t g = 0; g < G; g++) {
                for(ck_tile::index_t n = 0; n < N; n++) {
                    for(ck_tile::index_t c = 0; c < C; c++) {
                        for(ck_tile::index_t h = 0; h < H; h++) {
                            for(ck_tile::index_t w = 0; w < W; w++) {
                                ck_tile::index_t offset = g * in_strides[0] + n * in_strides[1] + 
                                                          c * in_strides[2] + h * in_strides[3] + w * in_strides[4];
                                in_ptr[offset] = static_cast<InDataType>(h);
                            }
                        }
                    }
                }
            }
            std::fill(wei_host.begin(), wei_host.end(), WeiDataType{1});
            std::cout << "Input: row pattern (h), Weight: all 1s" << std::endl;
        }
        break;
    case 5:
        // Column-wise (w % 64) + weight spatial pattern -- MATCHES ORIGINAL CK's init_input_data(true) + init_weight_data(true)
        {
            // Input: column-wise
            auto* in_ptr = in_host.data();
            for(ck_tile::index_t g = 0; g < G; g++) {
                for(ck_tile::index_t n = 0; n < N; n++) {
                    for(ck_tile::index_t c = 0; c < C; c++) {
                        for(ck_tile::index_t h = 0; h < H; h++) {
                            for(ck_tile::index_t w = 0; w < W; w++) {
                                ck_tile::index_t offset = g * in_strides[0] + n * in_strides[1] + 
                                                          c * in_strides[2] + h * in_strides[3] + w * in_strides[4];
                                in_ptr[offset] = static_cast<InDataType>(w % 64);
                            }
                        }
                    }
                }
            }
            // Weight: spatial pattern (fh * X + fw) -- matching original CK's init_weight_data(true)
            auto* wei_ptr = wei_host.data();
            for(ck_tile::index_t g = 0; g < G; g++) {
                for(ck_tile::index_t k = 0; k < K; k++) {
                    for(ck_tile::index_t c = 0; c < C; c++) {
                        for(ck_tile::index_t fh = 0; fh < Y; fh++) {
                            for(ck_tile::index_t fw = 0; fw < X; fw++) {
                                ck_tile::index_t offset = g * wei_strides[0] + k * wei_strides[1] + 
                                                          c * wei_strides[2] + fh * wei_strides[3] + fw * wei_strides[4];
                                wei_ptr[offset] = static_cast<WeiDataType>(fh * X + fw);
                            }
                        }
                    }
                }
            }
            std::cout << "Input: column-wise (w % 64), Weight: spatial pattern (fh*X+fw)" << std::endl;
        }
        break;
    default:
        // Random float values [-1, 1] (default, matching CK's GeneratorTensor_3)
        ck_tile::FillUniformDistribution<InDataType>{-1.f, 1.f}(in_host);
        ck_tile::FillUniformDistribution<WeiDataType>{-1.f, 1.f}(wei_host);
        break;
    }
    
    std::cout << "Init method: " << init_method 
              << " (0=zero, 1=int[-5,5], 2=rand, 3=col+w1, 4=row+w1, 5=col+wspatial)" << std::endl;

    // Compute CPU reference if verification is enabled
    if(do_verification)
    {
        depthwise_conv_fwd_cpu_reference<InDataType, WeiDataType, AccDataType, OutDataType>(
            in_host.data(),
            wei_host.data(),
            out_ref.data(),
            G, N, H, W, Ho, Wo, Y, X,
            stride_h, stride_w,
            dilation_h, dilation_w,
            pad_h, pad_w);
        std::cout << "Reference result computed on CPU" << std::endl;
    }

    // Allocate device memory
    ck_tile::DeviceMem in_dev(in_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_dev(wei_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_dev(out_host.get_element_space_size_in_bytes());

    // Copy to device
    in_dev.ToDevice(in_host.data());
    wei_dev.ToDevice(wei_host.data());

    // Prepare host arguments
    ck_tile::DepthwiseConvFwdHostArgs args;
    args.p_in  = in_dev.GetDeviceBuffer();
    args.p_wei = wei_dev.GetDeviceBuffer();
    args.p_out = out_dev.GetDeviceBuffer();
    args.G     = G;
    args.N     = N;
    args.C     = C;
    args.K     = K;
    args.Hi    = H;
    args.Wi    = W;
    args.Ho    = Ho;
    args.Wo    = Wo;
    args.Y     = Y;
    args.X     = X;
    args.stride_h   = stride_h;
    args.stride_w   = stride_w;
    args.dilation_h = dilation_h;
    args.dilation_w = dilation_w;
    args.pad_h      = pad_h;
    args.pad_w      = pad_w;

    // Reuse pre-calculated strides (avoid duplicate calculation)
    args.in_strides  = in_strides;
    args.wei_strides = wei_strides;
    args.out_strides = out_strides;

    // Stream config
    ck_tile::stream_config stream_cfg{
        nullptr,        // stream_id_
        true,           // time_kernel_
        1,              // log_level_
        warmup,         // cold_niters_
        repeat,         // nrepeat_
        true            // is_gpu_timer_
    };

    (void)timer_type;

    // Calculate performance metrics
    std::size_t flop     = conv_param.GetFlops();
    std::size_t num_byte = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    // Prepare verification info for invoker
    ck_tile::VerificationInfo<OutDataType> verify_info;
    verify_info.do_verification = do_verification;
    verify_info.p_out_dev = &out_dev;
    verify_info.p_out_host = &out_host;
    verify_info.p_out_ref = out_ref.data();
    verify_info.output_size = output_size;

    // Run kernel with verification
    auto [avg_time, best_config, is_verified, best_instance_idx] = 
        ck_tile::DepthwiseConvFwdInvoker::depthwise_conv_fwd<
            InDataType, WeiDataType, AccDataType, OutDataType>(args, stream_cfg, verify_info, flop, num_byte);

    if(avg_time < 0)
    {
        std::cerr << "Kernel launch failed\n";
        return -1;
    }

    float tflops    = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = static_cast<float>(num_byte) / 1.E6 / avg_time;

    // Print final result in CK-style format (matching original CK)
    // Example: [Instance 11] 0.414615 ms, 5.01887 TFlops, 1115.32 GB/s [PASS - Verified Correct], DeviceGroupedConvFwdDlV5<...>
    std::cout << "\n=== Best Performance ===" << std::endl;
    std::cout << "[Instance " << best_instance_idx << "] " 
              << std::fixed << std::setprecision(6) << avg_time << " ms, "
              << std::setprecision(5) << tflops << " TFlops, "
              << std::setprecision(3) << gb_per_sec << " GB/s";
    if(do_verification)
    {
        std::cout << " [" << (is_verified ? "PASS - Verified Correct" : "FAIL") << "]";
    }
    std::cout << ", " << best_config << std::endl;

    return is_verified ? 0 : 1;
}
