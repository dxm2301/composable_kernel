// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#define USE_V5_DEPTHWISE_CONV 1

// Template parameter explanation for DeviceGroupedConvFwdDlV5:
// 1. NDimSpatial         - Number of spatial dimensions (2 for 2D conv)
// 2. BlockSize           - Thread block size (64)
// 3. InKernelDataType    - Input data type (FP16/BF16/FP32/INT8)
// 4. WeiKernelDataType   - Weight data type
// 5. AccDataType         - Accumulator data type
// 6. OutKernelDataType   - Output data type
// 7. BlockTileSize       - Output tile size S<TileOutH, TileOutW>
// 8. FilterSize          - Convolution kernel size (3x3, 5x5, etc.)
// 9. FilterParam         - Tuple<Dilation, Stride, Padding>
//                          Example: Tuple<S<1,1>, S<2,2>, S<1,1>> means
//                          - Dilation: [1,1]
//                          - Stride: [2,2]
//                          - Padding: [1,1]
// 10-12. ElementOps      - Element-wise operations (In/Wei/Out)
// 13. NBatch             - Number of batches processed per block
// 14. SubTileH           - Sub-tile height per thread
// 15. SubTileW           - Sub-tile width per thread
// 16. InScalarPerVector  - Input vectorization width
// 17. OutScalarPerVector - Output vectorization width
// 18. RequirePadding     - Whether padding is required (usually false)

template <ck::index_t NDimSpatial>
using DeviceConvFwdFactory = std::tuple<
    // ==================== FilterSize = 5 ====================
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<14, 28>, 5, ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 2, 2, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 1, 1, 1, false>,

    // ==================== FilterSize = 3 ====================
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 3, ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8, 2, 2, 2, 2, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8, 2, 2, 2, 2, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 5, ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4, 2, 2, 2, 2, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 3, ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 8, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 3, ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<14, 28>, 3, ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1, 2, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 32>, 5, ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4, 1, 8, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 3, ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 8, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 3, ck::Tuple<S<1,1>, S<1,1>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 3, ck::Tuple<S<1,1>, S<2,2>, S<1,1>>, InElementOp, WeiElementOp, OutElementOp, 2, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 5, ck::Tuple<S<1,1>, S<1,1>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 4, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 5, ck::Tuple<S<1,1>, S<2,2>, S<2,2>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,

    // ==================== FilterSize = 7 (padding = 3) ====================
    // Small tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 1, 1, 1, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 8, 2, 2, 2, 2, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 4, 2, 2, 2, 2, false>,
    // Medium tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 8, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 2, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<14, 28>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 2, 2, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 32>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 4, 1, 8, 8, 8, false>,
    // Large tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 8, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 7, ck::Tuple<S<1,1>, S<1,1>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 4, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 2, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 7, ck::Tuple<S<1,1>, S<2,2>, S<3,3>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,

    // ==================== FilterSize = 9 (padding = 4) ====================
    // Small tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 1, 1, 1, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 8, 2, 2, 2, 2, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<8, 8>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 4, 2, 2, 2, 2, false>,
    // Medium tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 1, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 8, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 16>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 2, 1, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<14, 28>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 2, 2, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<16, 32>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 4, 1, 8, 8, 8, false>,
    // Large tiles
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<28, 28>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 8, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 9, ck::Tuple<S<1,1>, S<1,1>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 4, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 2, 4, 4, 8, 8, false>,
    ck::tensor_operation::device::DeviceGroupedConvFwdDlV5<NDimSpatial, 64, InKernelDataType, WeiKernelDataType, AccDataType, OutKernelDataType, S<32, 32>, 9, ck::Tuple<S<1,1>, S<2,2>, S<4,4>>, InElementOp, WeiElementOp, OutElementOp, 1, 4, 4, 8, 8, false>
>;

template <ck::index_t NDimSpatial>
using HostConvFwdInstance = ck::tensor_operation::host::ReferenceConvFwd<NDimSpatial,
                                                                         InUserDataType,
                                                                         WeiUserDataType,
                                                                         CShuffleDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         PassThrough>;

template <ck::index_t NDimSpatial>
bool run_depthwise_conv_fwd(const ExecutionConfig& config,
                          const ck::utils::conv::ConvParam& conv_param)
{
    static_assert(1 <= NDimSpatial && NDimSpatial <= 3, "Unsupported NDimSpatial");
    
    const bool debug_mode = config.is_debug_mode();
    
    const auto in_g_n_c_wis_desc  = make_input_descriptor(conv_param);
    const auto wei_g_k_c_xs_desc  = make_weight_descriptor(conv_param);
    const auto out_g_n_k_wos_desc = make_output_descriptor(conv_param);

    Tensor<InUserDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiUserDataType> wei(wei_g_k_c_xs_desc);
    Tensor<OutUserDataType> out_host(out_g_n_k_wos_desc);
    Tensor<OutKernelDataType> out_device(out_g_n_k_wos_desc);

    // Debug mode: print detailed tensor info
    if(debug_mode)
    {
        std::cout << "\n=== Debug Info ===" << std::endl;
        std::cout << "in: " << in.mDesc << std::endl;
        std::cout << "wei: " << wei.mDesc << std::endl;
        std::cout << "out: " << out_host.mDesc << std::endl;
    }

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InUserDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiUserDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InUserDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiUserDataType>{-0.5, 0.5});
    }

    DeviceMem in_device_buf(sizeof(InKernelDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiKernelDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutKernelDataType) * out_device.mDesc.GetElementSpaceSize());
    
    // Debug mode: print element sizes
    if(debug_mode)
    {
        std::cout << "in elem_size: " << in.mDesc.GetElementSpaceSize() << std::endl;
        std::cout << "wei elem_size: " << wei.mDesc.GetElementSpaceSize() << std::endl;
        std::cout << "out elem_size: " << out_device.mDesc.GetElementSpaceSize() << std::endl;
    }

#ifdef BUILD_INT4_EXAMPLE
    const Tensor<InKernelDataType> in_converted(in);
    const Tensor<WeiKernelDataType> wei_converted(wei);

    in_device_buf.ToDevice(in_converted.mData.data());
    wei_device_buf.ToDevice(wei_converted.mData.data());
#else
    in_device_buf.ToDevice(in.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());
#endif

    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> a_g_n_c_wis_strides{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> b_g_k_c_xs_strides{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> e_g_n_k_wos_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(in_g_n_c_wis_desc.GetLengths(), a_g_n_c_wis_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), a_g_n_c_wis_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), b_g_k_c_xs_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), b_g_k_c_xs_strides);
    copy(out_g_n_k_wos_desc.GetLengths(), e_g_n_k_wos_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), e_g_n_k_wos_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    std::size_t flop = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InUserDataType, WeiUserDataType, OutUserDataType>();
    
    float best_time = std::numeric_limits<float>::max();
    int best_idx = -1;
    std::string best_type_string;
    
    // Prepare reference result if verification is enabled
    if(config.do_verification)
    {
        auto ref_conv     = HostConvFwdInstance<NDimSpatial>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei,
                                                  out_host,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{});
        ref_invoker.Run(ref_argument);
        if(debug_mode) std::cout << "Reference result computed on CPU" << std::endl;
    }
    
    if(debug_mode) std::cout << "\n=== Testing all instances ===" << std::endl;
    
    ck::static_for<0, std::tuple_size_v<DeviceConvFwdFactory<NDimSpatial>>, 1>{}([&](auto i) {
        using ConvInstance = std::tuple_element_t<i, DeviceConvFwdFactory<NDimSpatial>>;
        auto conv = ConvInstance{};
        auto argument = conv.MakeArgument(in_device_buf.GetDeviceBuffer(),
                                         wei_device_buf.GetDeviceBuffer(),
                                         std::array<const void*, 0>{},
                                         out_device_buf.GetDeviceBuffer(),
                                         a_g_n_c_wis_lengths,
                                         a_g_n_c_wis_strides,
                                         b_g_k_c_xs_lengths,
                                         b_g_k_c_xs_strides,
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         std::array<std::array<ck::index_t, NDimSpatial + 3>, 0>{},
                                         e_g_n_k_wos_lengths,
                                         e_g_n_k_wos_strides,
                                         conv_filter_strides,
                                         conv_filter_dilations,
                                         input_left_pads,
                                         input_right_pads,
                                         InElementOp{},
                                         WeiElementOp{},
                                         OutElementOp{});
        
        if(conv.IsSupportedArgument(argument))
        {
            auto invoker = conv.MakeInvoker();
            float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
            
            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;
            float gb_per_sec = num_btype / 1.E6 / avg_time;
            
            bool is_correct = true;
            std::string status = "";
            
            // Verify correctness immediately after running
            if(config.do_verification)
            {
                out_device_buf.FromDevice(out_device.mData.data());
                
#ifdef BUILD_INT4_EXAMPLE
                const Tensor<OutUserDataType> out_device_converted(out_device);
                is_correct = ck::utils::check_err(
                    out_device_converted.mData, out_host.mData, "", 1e-5f, 1e-4f);
#else
                is_correct = ck::utils::check_err(
                    out_device.mData, out_host.mData, "", 1e-5f, 1e-4f);
#endif
                status = is_correct ? " [PASS]" : " [FAIL]";
            }
            
            // Debug mode: print each instance result
            if(debug_mode)
            {
                std::cout << "[Instance " << i << "]" << status << " " 
                          << avg_time << " ms, " << tflops << " TFlops, " 
                          << gb_per_sec << " GB/s, " << conv.GetTypeString() << std::endl;
            }
            
            // Only consider correct results for best performance
            if(is_correct && avg_time < best_time) {
                best_time = avg_time;
                best_idx = i;
                best_type_string = conv.GetTypeString();
            }
        }
    });
    
    if(best_idx < 0)
    {
        std::string error_msg = config.do_verification 
            ? "No correct instance found for the specified Conv parameters"
            : "No matching instance found for the specified Conv parameters";
        throw std::runtime_error(error_msg);
    }
    
    float best_tflops = static_cast<float>(flop) / 1.E9 / best_time;
    float best_gb_per_sec = num_btype / 1.E6 / best_time;
    float best_gflops = best_tflops * 1000.0f;  // TFlops -> GFlops for MIOpen format

    // Debug mode: print detailed best performance
    if(debug_mode)
    {
        std::cout << "\n=== Best Performance ===" << std::endl;
        std::cout << "[Instance " << best_idx << "] " 
                 << best_time << " ms, " << best_tflops << " TFlops, " 
                 << best_gb_per_sec << " GB/s";
        if(config.do_verification)
        {
            std::cout << " [PASS - Verified Correct]";
        }
        else
        {
            std::cout << " [NOT VERIFIED]";
        }
        std::cout << ", " << best_type_string << std::endl;
    }

    // Extract parameters from conv_param (for stats output)
    const auto G = conv_param.G_;
    const auto N = conv_param.N_;
    const auto Y = conv_param.filter_spatial_lengths_[0];
    const auto X = conv_param.filter_spatial_lengths_[1];
    const auto Ho = conv_param.output_spatial_lengths_[0];
    const auto Wo = conv_param.output_spatial_lengths_[1];
    const auto stride_h = conv_param.conv_filter_strides_[0];

    // Calculate bytes read/written for stats line
    std::size_t bytes_read = (in.mDesc.GetElementSpaceSize() + wei.mDesc.GetElementSpaceSize()) * sizeof(InKernelDataType);
    std::size_t bytes_written = out_device.mDesc.GetElementSpaceSize() * sizeof(OutKernelDataType);

    // Build conv name for stats line
    std::ostringstream conv_name;
    conv_name << "fwd-conv" << Y << "x" << X << "u" << stride_h;

    // Print CK algorithm/solution info (MIOpen-compatible)
    std::cout << "CK Forward Conv. Solution: Instance " << best_idx 
              << ", " << best_type_string << std::endl;

    // Print GPU kernel time (MIOpen-compatible)
    std::cout << "GPU Kernel Time Forward Conv. Elapsed: " 
              << std::fixed << std::setprecision(6) << best_time << " ms (best)" << std::endl;

    // Print stats header and data (MIOpen-compatible format)
    std::cout << "stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs" << std::endl;
    std::cout << "stats: " << conv_name.str() << ", "
              << N << ", " << G << ", " << Ho << ", " << Wo << ", "
              << X << ", " << Y << ", " << G << ", "
              << flop << ", " << bytes_read << ", " << bytes_written << ", "
              << std::fixed << std::setprecision(0) << best_gflops << ", "
              << std::setprecision(0) << best_gb_per_sec << ", "
              << std::setprecision(6) << best_time << std::endl;

    // Print verification result (MIOpen-compatible format)
    if(config.do_verification)
    {
        std::cout << "Forward Convolution Verifies OK on GPU reference (instance=" 
                  << best_idx << ")" << std::endl;
    }

    return true;
}

bool run_depthwise_conv_fwd_example(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return false;
    }

    // Print command line (always shown)
    for(int i = 0; i < argc; ++i)
    {
        if(i > 0) std::cout << " ";
        std::cout << argv[i];
    }
    std::cout << std::endl;

    // Print environment info (MIOpen-compatible)
    // PRNG seed: use MIOPEN_DEBUG_PRNG_SEED env var if set, otherwise time-based
    std::cout << "PRNG seed: " << SystemInfo::get_prng_seed() << std::endl;
    std::cout << SystemInfo::get_full_env_string() << std::endl;

    switch(conv_param.num_dim_spatial_)
    {
    case 1: 
        std::cout << "1D depthwise conv not supported by v4 implementation, only 2D" << std::endl;
        return false;
    case 2: return run_depthwise_conv_fwd<2>(config, conv_param);
    case 3: 
        std::cout << "3D depthwise conv not supported by v4 implementation, only 2D" << std::endl;
        return false;
    }

    return false;
}
