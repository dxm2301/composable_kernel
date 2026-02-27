// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CPU reference for depthwise convolution forward. C=K=1 per group.
//   Input:  [G][N][1][Hi][Wi]  strides [Hi*Wi, G*Hi*Wi, 1, Wi, 1]
//   Weight: [G][1][1][Y][X]    strides [Y*X,   G*Y*X,   1, X,  1]
//   Output: [G][N][1][Ho][Wo]  strides [Ho*Wo, G*Ho*Wo, 1, Wo, 1]
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
    // int64_t strides to avoid overflow with large tensors
    const int64_t in_g_stride = static_cast<int64_t>(Hi) * Wi;
    const int64_t in_n_stride = static_cast<int64_t>(G) * Hi * Wi;
    const int64_t in_h_stride = Wi;
    const int64_t in_w_stride = 1;

    const int64_t wei_g_stride = static_cast<int64_t>(Y) * X;
    const int64_t wei_y_stride = X;
    const int64_t wei_x_stride = 1;

    const int64_t out_g_stride = static_cast<int64_t>(Ho) * Wo;
    const int64_t out_n_stride = static_cast<int64_t>(G) * Ho * Wo;
    const int64_t out_h_stride = Wo;
    const int64_t out_w_stride = 1;

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
                                int64_t in_idx = static_cast<int64_t>(g) * in_g_stride +
                                                 static_cast<int64_t>(n) * in_n_stride +
                                                 static_cast<int64_t>(hi) * in_h_stride +
                                                 static_cast<int64_t>(wi) * in_w_stride;
                                int64_t wei_idx = static_cast<int64_t>(g) * wei_g_stride +
                                                  static_cast<int64_t>(y) * wei_y_stride +
                                                  static_cast<int64_t>(x) * wei_x_stride;

                                acc += static_cast<AccDataType>(p_in[in_idx]) *
                                       static_cast<AccDataType>(p_wei[wei_idx]);
                            }
                        }
                    }

                    int64_t out_idx = static_cast<int64_t>(g) * out_g_stride +
                                      static_cast<int64_t>(n) * out_n_stride +
                                      static_cast<int64_t>(ho) * out_h_stride +
                                      static_cast<int64_t>(wo) * out_w_stride;
                    p_out[out_idx] = static_cast<OutDataType>(acc);
                }
            }
        }
    }
}

template <typename InDataType,
          typename WeiDataType,
          typename AccDataType,
          typename OutDataType>
int run_depthwise_conv_fwd_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const int verbosity           = arg_parser.get_int("v");
    const bool do_verification    = (verbosity > 0);
    const bool debug_mode         = (verbosity >= 2);
    const int warmup              = arg_parser.get_int("warmup");
    const int repeat              = arg_parser.get_int("repeat");
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

    constexpr ck_tile::index_t C = 1;
    constexpr ck_tile::index_t K = 1;

    DepthwiseConvParam conv_param(
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

    const ck_tile::index_t Ho = conv_param.output_spatial_lengths_[0];
    const ck_tile::index_t Wo = conv_param.output_spatial_lengths_[1];

    std::string prec_str;
    if constexpr(std::is_same_v<InDataType, ck_tile::fp16_t>)
        prec_str = "fp16";
    else if constexpr(std::is_same_v<InDataType, ck_tile::bf16_t>)
        prec_str = "bf16";
    else
        prec_str = "fp32";

    const auto in_strides  = GetGNCHWStrides(G, N, C, H, W);
    const auto wei_strides = GetGKCYXStrides(G, K, C, Y, X);
    const auto out_strides = GetGNCHWStrides(G, N, K, Ho, Wo);

    std::cout << "./bin/tile_example_depthwise_conv_fwd"
              << " -G=" << G << " -N=" << N << " -H=" << H << " -W=" << W
              << " -Y=" << Y << " -X=" << X
              << " -stride_h=" << stride_h << " -stride_w=" << stride_w
              << " -dilation_h=" << dilation_h << " -dilation_w=" << dilation_w
              << " -pad_h=" << pad_h << " -pad_w=" << pad_w
              << " -prec=" << prec_str
              << " -v=" << verbosity << std::endl;

    std::cout << "PRNG seed: " << ck_tile::SystemInfo::get_prng_seed() << std::endl;
    std::cout << ck_tile::SystemInfo::get_full_env_string() << std::endl;

    if(debug_mode)
    {
        std::cout << "\n=== Debug Info ===" << std::endl;
        std::cout << "in: dim 5, lengths {" << G << ", " << N << ", " << C << ", " << H << ", " << W << "}, "
                  << "strides {" << in_strides[0] << ", " << in_strides[1] << ", " << in_strides[2]
                  << ", " << in_strides[3] << ", " << in_strides[4] << "}" << std::endl;
        std::cout << "wei: dim 5, lengths {" << G << ", " << K << ", " << C << ", " << Y << ", " << X << "}, "
                  << "strides {" << wei_strides[0] << ", " << wei_strides[1] << ", " << wei_strides[2]
                  << ", " << wei_strides[3] << ", " << wei_strides[4] << "}" << std::endl;
        std::cout << "out: dim 5, lengths {" << G << ", " << N << ", " << K << ", " << Ho << ", " << Wo << "}, "
                  << "strides {" << out_strides[0] << ", " << out_strides[1] << ", " << out_strides[2]
                  << ", " << out_strides[3] << ", " << out_strides[4] << "}" << std::endl;
    }

    const std::size_t input_size = static_cast<std::size_t>(G) * N * C * H * W;
    const std::size_t weight_size = static_cast<std::size_t>(G) * K * C * Y * X;
    const std::size_t output_size = static_cast<std::size_t>(G) * N * K * Ho * Wo;

    if(debug_mode)
    {
        std::cout << "in elem_count: " << input_size << std::endl;
        std::cout << "wei elem_count: " << weight_size << std::endl;
        std::cout << "out elem_count: " << output_size << std::endl;
    }

    ck_tile::HostTensor<InDataType> in_host({input_size});
    ck_tile::HostTensor<WeiDataType> wei_host({weight_size});
    ck_tile::HostTensor<OutDataType> out_host({output_size});
    ck_tile::HostTensor<OutDataType> out_ref({output_size});

    auto fill_input_pattern = [&](auto value_fn) {
        auto* ptr = in_host.data();
        for(ck_tile::index_t g = 0; g < G; g++)
            for(ck_tile::index_t n = 0; n < N; n++)
                for(ck_tile::index_t c = 0; c < C; c++)
                    for(ck_tile::index_t h = 0; h < H; h++)
                        for(ck_tile::index_t w = 0; w < W; w++)
                        {
                            ck_tile::index_t offset = g * in_strides[0] + n * in_strides[1] +
                                                      c * in_strides[2] + h * in_strides[3] + w * in_strides[4];
                            ptr[offset] = static_cast<InDataType>(value_fn(h, w));
                        }
    };

    switch(init_method)
    {
    case 0:
        std::fill(in_host.begin(), in_host.end(), InDataType{0});
        std::fill(wei_host.begin(), wei_host.end(), WeiDataType{0});
        break;
    case 1:
        ck_tile::FillUniformDistributionIntegerValue<InDataType>{-5.f, 5.f}(in_host);
        ck_tile::FillUniformDistributionIntegerValue<WeiDataType>{-5.f, 5.f}(wei_host);
        break;
    case 3:
        fill_input_pattern([](ck_tile::index_t, ck_tile::index_t w) { return w % 64; });
        std::fill(wei_host.begin(), wei_host.end(), WeiDataType{1});
        if(debug_mode)
        {
            std::cout << "Input: column-wise (w % 64), Weight: all 1s" << std::endl;
        }
        break;
    case 4:
        fill_input_pattern([](ck_tile::index_t h, ck_tile::index_t) { return h; });
        std::fill(wei_host.begin(), wei_host.end(), WeiDataType{1});
        if(debug_mode)
        {
            std::cout << "Input: row pattern (h), Weight: all 1s" << std::endl;
        }
        break;
    case 5:
        {
            fill_input_pattern([](ck_tile::index_t, ck_tile::index_t w) { return w % 64; });
            auto* wei_ptr = wei_host.data();
            for(ck_tile::index_t g = 0; g < G; g++)
                for(ck_tile::index_t k = 0; k < K; k++)
                    for(ck_tile::index_t c = 0; c < C; c++)
                        for(ck_tile::index_t fh = 0; fh < Y; fh++)
                            for(ck_tile::index_t fw = 0; fw < X; fw++)
                            {
                                ck_tile::index_t offset = g * wei_strides[0] + k * wei_strides[1] +
                                                          c * wei_strides[2] + fh * wei_strides[3] + fw * wei_strides[4];
                                wei_ptr[offset] = static_cast<WeiDataType>(fh * X + fw);
                            }
            if(debug_mode)
            {
                std::cout << "Input: column-wise (w % 64), Weight: spatial pattern (fh*X+fw)" << std::endl;
            }
        }
        break;
    default: // covers init_method=2 and any unlisted values
        ck_tile::FillUniformDistribution<InDataType>{-1.f, 1.f}(in_host);
        ck_tile::FillUniformDistribution<WeiDataType>{-1.f, 1.f}(wei_host);
        break;
    }

    if(debug_mode)
    {
        std::cout << "Init method: " << init_method
                  << " (0=zero, 1=int[-5,5], 2=rand, 3=col+w1, 4=row+w1, 5=col+wspatial)" << std::endl;
    }

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
        if(debug_mode)
        {
            std::cout << "Reference result computed on CPU" << std::endl;
        }
    }

    ck_tile::DeviceMem in_dev(in_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem wei_dev(wei_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem out_dev(out_host.get_element_space_size_in_bytes());

    in_dev.ToDevice(in_host.data());
    wei_dev.ToDevice(wei_host.data());

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

    args.in_strides  = in_strides;
    args.wei_strides = wei_strides;
    args.out_strides = out_strides;

    ck_tile::stream_config stream_cfg{
        nullptr, true, debug_mode ? 1 : 0, warmup, repeat, true};

    std::size_t flop     = conv_param.GetFlops();
    std::size_t num_byte = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    VerificationInfo<OutDataType> verify_info;
    verify_info.do_verification = do_verification;
    verify_info.p_out_dev = &out_dev;
    verify_info.p_out_host = &out_host;
    verify_info.p_out_ref = out_ref.data();
    verify_info.output_size = output_size;

    auto [avg_time, best_config, is_verified, best_instance_idx] =
        DepthwiseConvFwdInvoker::depthwise_conv_fwd<
            InDataType, WeiDataType, AccDataType, OutDataType>(args, stream_cfg, verify_info, flop, num_byte);

    if(avg_time < 0)
    {
        std::cerr << "Kernel launch failed" << std::endl;
        return 1;
    }

    float tflops    = static_cast<float>(flop) / 1.E9 / avg_time;
    float gb_per_sec = static_cast<float>(num_byte) / 1.E6 / avg_time;
    float gflops    = tflops * 1000.0f;  // TFlops -> GFlops for MIOpen format

    if(debug_mode)
    {
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
    }

    std::size_t bytes_read = input_size * sizeof(InDataType) + weight_size * sizeof(WeiDataType);
    std::size_t bytes_written = output_size * sizeof(OutDataType);

    std::ostringstream conv_name;
    conv_name << "fwd-conv" << Y << "x" << X << "u" << stride_h;

    std::cout << "CK Forward Conv. Solution: Instance " << best_instance_idx
              << ", " << best_config << std::endl;

    std::cout << "GPU Kernel Time Forward Conv. Elapsed: "
              << std::fixed << std::setprecision(6) << avg_time << " ms (best)" << std::endl;

    // MIOpen-compatible stats format: c=G, k=G for depthwise (C=K=1 per group)
    std::cout << "stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs" << std::endl;
    std::cout << "stats: " << conv_name.str() << ", "
              << N << ", " << G << ", " << Ho << ", " << Wo << ", "
              << X << ", " << Y << ", " << G << ", "
              << flop << ", " << bytes_read << ", " << bytes_written << ", "
              << std::fixed << std::setprecision(0) << gflops << ", "
              << std::setprecision(0) << gb_per_sec << ", "
              << std::setprecision(6) << avg_time << std::endl;

    if(do_verification)
    {
        if(is_verified)
        {
            std::cout << "Forward Convolution Verifies OK against CPU reference (instance="
                      << best_instance_idx << ")" << std::endl;
        }
        else
        {
            std::cout << "Forward Convolution FAILED (instance=" << best_instance_idx << ")" << std::endl;
        }
    }

    return is_verified ? 0 : 1;
}
