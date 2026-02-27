// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifndef __HIPCC_RTC__
// Provides <string>, <hip/hip_runtime.h>
#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/host/hip_check_error.hpp"

#include <iomanip>
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <random>

#ifdef __linux__
#include <unistd.h>
#include <sys/utsname.h>
#endif

namespace ck_tile {

/// System information utilities for benchmark logging (MIOpen-compatible format)
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
        std::string name = get_device_name();
        return name.empty() ? "unknown" : name;
    }

    static int get_gpu_count()
    {
        int count = 0;
        (void)hipGetDeviceCount(&count);  // Ignore error, return 0 on failure
        return count;
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

} // namespace ck_tile

#endif // __HIPCC_RTC__

