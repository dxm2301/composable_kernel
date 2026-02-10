// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <type_traits>

// Additional headers for SystemInfo (timestamp, hostname, OS info)
#include <sstream>
#include <chrono>
#include <ctime>
#include <cstdlib>  // For std::getenv, std::strtoul
#include <random>   // For std::random_device

#ifdef __linux__
#include <unistd.h>
#include <sys/utsname.h>
#endif
// Note: <iomanip> is included via "ck/library/utility/check_err.hpp"

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_v4.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_v5.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"

using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using FP32 = float;
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
using I4 = ck::int4_t;
#endif
using I8  = std::int8_t;
using I32 = std::int32_t;

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <typename InputLay, typename WeightLay, typename OutputLay>
struct CommonLayoutSetting
{
    using InputLayout  = InputLay;
    using WeightLayout = WeightLay;
    using OutputLayout = OutputLay;
};

template <ck::index_t NDimSpatial>
struct CommonLayoutSettingSelector;

namespace ctl = ck::tensor_layout::convolution;

// template <>
// struct CommonLayoutSettingSelector<1> final
//     : CommonLayoutSetting<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>
// {
// };

template <>
struct CommonLayoutSettingSelector<2> final
    // : CommonLayoutSetting<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>
    : CommonLayoutSetting<ctl::GNCHW, ctl::GKCYX, ctl::GNKHW>
{
};

// template <>
// struct CommonLayoutSettingSelector<3> final
//     : CommonLayoutSetting<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>
// {
// };

template <ck::index_t NDimSpatial>
using InputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::InputLayout;

template <ck::index_t NDimSpatial>
using WeightLayout = typename CommonLayoutSettingSelector<NDimSpatial>::WeightLayout;

template <ck::index_t NDimSpatial>
using OutputLayout = typename CommonLayoutSettingSelector<NDimSpatial>::OutputLayout;

// System information utilities for benchmark logging (MIOpen-compatible)
struct SystemInfo
{
    static std::string get_timestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm* gmt = std::gmtime(&time_t_now);
        std::ostringstream oss;
        oss << std::put_time(gmt, "%Y-%m-%d %H:%M:%S") << " UTC";
        return oss.str();
    }

    static std::string get_hostname()
    {
#ifdef __linux__
        char hostname[256];
        if(gethostname(hostname, sizeof(hostname)) == 0)
        {
            return std::string(hostname);
        }
#endif
        return "unknown";
    }

    static std::string get_os_info()
    {
#ifdef __linux__
        struct utsname buf;
        if(uname(&buf) == 0)
        {
            return std::string(buf.sysname) + " " + std::string(buf.release);
        }
#endif
        return "unknown";
    }

    static std::string get_rocm_version()
    {
        std::ostringstream oss;
#if defined(HIP_VERSION_MAJOR) && defined(HIP_VERSION_MINOR) && defined(HIP_VERSION_PATCH)
        oss << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "." << HIP_VERSION_PATCH;
#else
        oss << "unknown";
#endif
        return oss.str();
    }

    static std::string get_gpu_name()
    {
        hipDeviceProp_t props{};
        int device;
        if(hipGetDevice(&device) != hipSuccess)
        {
            return "unknown";
        }
        if(hipGetDeviceProperties(&props, device) != hipSuccess)
        {
            return "unknown";
        }
        return std::string(props.name);
    }

    static std::string get_full_env_string()
    {
        std::ostringstream oss;
        oss << "Timestamp: " << get_timestamp() << "; "
            << "Host Name: " << get_hostname() << "; "
            << "OS: " << get_os_info() << "; "
            << "ROCm: " << get_rocm_version() << "; "
            << "GPU: " << get_gpu_name();
        return oss.str();
    }

    // PRNG seed: uses MIOPEN_DEBUG_DRIVER_PRNG_SEED env var (default: 12345678, 0 = random)
    static unsigned int get_prng_seed()
    {
        static unsigned int seed = [] {
            const char* env_seed = std::getenv("MIOPEN_DEBUG_DRIVER_PRNG_SEED");
            unsigned int external_seed = 12345678u;  // Default like MIOpen
            
            if(env_seed != nullptr)
            {
                external_seed = static_cast<unsigned int>(std::strtoul(env_seed, nullptr, 10));
            }
            
            // If env var is 0, use random device; otherwise use env var value (or default)
            if(external_seed == 0)
            {
                std::random_device rd;
                return rd();
            }
            return external_seed;
        }();
        return seed;
    }
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = true;
    int verbosity        = 1;  // 0=no verify, 1=standard, 2=debug
    
    bool is_debug_mode() const { return verbosity >= 2; }
};

#define DefaultConvParam                                                       \
    ck::utils::conv::ConvParam                                                 \
    {                                                                          \
        2, 32, 128, 1, 1, {3, 3}, {32, 64}, {1, 1}, {1, 1}, {1, 1}, { 1, 1 } \
    }//                                      s0 s1   d0 d1   t  l     d  r

inline void print_help_msg()
{
    std::cerr << "arg1: verification/verbosity (0=no verify, 1=standard mode, 2=debug mode)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

inline bool parse_cmd_args(int argc,
                           char* argv[],
                           ExecutionConfig& config,
                           ck::utils::conv::ConvParam& conv_param)
{
    constexpr int num_execution_config_args =
        3; // arguments for do_verification, init_method, time_kernel
    constexpr int num_conv_param_leading_args = 5; // arguments for num_dim_spatial_, G_, N_, K_, C_

    constexpr int threshold_to_catch_partial_args = 1 + num_execution_config_args;
    constexpr int threshold_to_catch_all_args =
        threshold_to_catch_partial_args + num_conv_param_leading_args;

    if(argc == 1)
    {
        // use default
    }
    // catch only ExecutionConfig arguments
    else if(argc == threshold_to_catch_partial_args)
    {
        config.verbosity       = std::stoi(argv[1]);
        config.do_verification = (config.verbosity > 0);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);
    }
    // catch both ExecutionConfig & ConvParam arguments
    else if(threshold_to_catch_all_args < argc && ((argc - threshold_to_catch_all_args) % 3 == 0))
    {
        config.verbosity       = std::stoi(argv[1]);
        config.do_verification = (config.verbosity > 0);
        config.init_method     = std::stoi(argv[2]);
        config.time_kernel     = std::stoi(argv[3]);

        const ck::index_t num_dim_spatial = std::stoi(argv[4]);
        conv_param                        = ck::utils::conv::parse_conv_param(
            num_dim_spatial, threshold_to_catch_partial_args + 1, argv);
    }
    else
    {
        print_help_msg();
        return false;
    }

    return true;
}

inline HostTensorDescriptor make_input_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.C_, conv_param.input_spatial_lengths_[0]},
            {
                conv_param.C_,                                                        // g
                conv_param.input_spatial_lengths_[0] * conv_param.G_ * conv_param.C_, // n
                1,                                                                    // c
                conv_param.G_ * conv_param.C_                                         // wi
            },
            ck::tensor_layout::BypassLayoutVerification{});

    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1]},
            {
                conv_param.C_ * conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1], // g
                conv_param.G_ * conv_param.C_ * conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1], // n
                // conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1], // c
                1, // c
                conv_param.input_spatial_lengths_[1] * conv_param.C_, // hi
                conv_param.C_                                      // wi
            },
            ck::tensor_layout::BypassLayoutVerification{});

    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.C_,
             conv_param.input_spatial_lengths_[0],
             conv_param.input_spatial_lengths_[1],
             conv_param.input_spatial_lengths_[2]},
            {
                conv_param.C_, // g
                conv_param.input_spatial_lengths_[0] * conv_param.input_spatial_lengths_[1] *
                    conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // n
                1,                                                                        // c
                conv_param.input_spatial_lengths_[1] * conv_param.input_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.C_,                                    // di
                conv_param.input_spatial_lengths_[2] * conv_param.G_ * conv_param.C_, // hi
                conv_param.G_ * conv_param.C_                                         // wi
            },
            ck::tensor_layout::BypassLayoutVerification{});
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_weight_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.K_, conv_param.C_, conv_param.filter_spatial_lengths_[0]},
            {
                conv_param.K_ * conv_param.filter_spatial_lengths_[0] * conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.C_,                 // k
                1,                                                                     // c
                conv_param.C_                                                          // x
            },
            ck::tensor_layout::BypassLayoutVerification{});
    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1]},
            {
                conv_param.C_ * conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1], // g
                conv_param.G_ * conv_param.C_ * conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1], // k
                // conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1], // c
                1, // c
                conv_param.filter_spatial_lengths_[1], // y
                1                                       // x
            },
            ck::tensor_layout::BypassLayoutVerification{});
    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.K_,
             conv_param.C_,
             conv_param.filter_spatial_lengths_[0],
             conv_param.filter_spatial_lengths_[1],
             conv_param.filter_spatial_lengths_[2]},
            {
                conv_param.K_ * conv_param.filter_spatial_lengths_[0] *
                    conv_param.filter_spatial_lengths_[1] * conv_param.filter_spatial_lengths_[2] *
                    conv_param.C_, // g
                conv_param.filter_spatial_lengths_[0] * conv_param.filter_spatial_lengths_[1] *
                    conv_param.filter_spatial_lengths_[2] * conv_param.C_, // k
                1,                                                         // c
                conv_param.filter_spatial_lengths_[1] * conv_param.filter_spatial_lengths_[2] *
                    conv_param.C_,                                     // z
                conv_param.filter_spatial_lengths_[2] * conv_param.C_, // y
                conv_param.C_                                          // x
            },
            ck::tensor_layout::BypassLayoutVerification{});
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_bias_descriptor(const ck::utils::conv::ConvParam& conv_param)
{
    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_, // g
                0,             // k
                1,             // c
                0              // x
            },
            ck::tensor_layout::BypassLayoutVerification{});
    case 2:
        return HostTensorDescriptor({conv_param.G_,
                                     conv_param.N_,
                                     conv_param.K_,
                                     conv_param.output_spatial_lengths_[0],
                                     conv_param.output_spatial_lengths_[1]},
                                    {
                                        conv_param.K_, // g
                                        0,             // n
                                        1,             // k
                                        0,             // ho
                                        0              // wo
                                    },
                                    ck::tensor_layout::BypassLayoutVerification{});
    case 3:
        return HostTensorDescriptor({conv_param.G_,
                                     conv_param.N_,
                                     conv_param.K_,
                                     conv_param.output_spatial_lengths_[0],
                                     conv_param.output_spatial_lengths_[1],
                                     conv_param.output_spatial_lengths_[2]},
                                    {
                                        conv_param.K_, // g
                                        0,             // n
                                        1,             // k
                                        0,             // z
                                        0,             // y
                                        0              // x
                                    },
                                    ck::tensor_layout::BypassLayoutVerification{});
    }

    throw std::runtime_error("unsuppored # dim spatial");
}

inline HostTensorDescriptor make_output_descriptor(const ck::utils::conv::ConvParam& conv_param)
{

    switch(conv_param.num_dim_spatial_)
    {
    case 1:
        return HostTensorDescriptor(
            {conv_param.G_, conv_param.N_, conv_param.K_, conv_param.output_spatial_lengths_[0]},
            {
                conv_param.K_,                                                         // g
                conv_param.output_spatial_lengths_[0] * conv_param.G_ * conv_param.K_, // n
                1,                                                                     // k
                conv_param.G_ * conv_param.K_                                          // wo
            },
            ck::tensor_layout::BypassLayoutVerification{});
    case 2:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1]},
            {
                conv_param.K_ * conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1], // g
                conv_param.G_ * conv_param.K_ * conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1], // n
                // conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1], // k
                1, // c
                conv_param.output_spatial_lengths_[1] * conv_param.C_, // ho
                conv_param.C_                                       // wo
            },
            ck::tensor_layout::BypassLayoutVerification{});

    case 3:
        return HostTensorDescriptor(
            {conv_param.G_,
             conv_param.N_,
             conv_param.K_,
             conv_param.output_spatial_lengths_[0],
             conv_param.output_spatial_lengths_[1],
             conv_param.output_spatial_lengths_[2]},
            {
                conv_param.K_, // g
                conv_param.output_spatial_lengths_[0] * conv_param.output_spatial_lengths_[1] *
                    conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // n
                1,                                                                         // k
                conv_param.output_spatial_lengths_[1] * conv_param.output_spatial_lengths_[2] *
                    conv_param.G_ * conv_param.K_,                                     // do
                conv_param.output_spatial_lengths_[2] * conv_param.G_ * conv_param.K_, // ho
                conv_param.G_ * conv_param.K_                                          // wo
            },
            ck::tensor_layout::BypassLayoutVerification{});
    }

    throw std::runtime_error("unsuppored # dim spatial");
}
