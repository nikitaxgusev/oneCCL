/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>

#include "common/api_wrapper/api_wrapper.hpp"
#include "common/log/log.hpp"
#include "common/api_wrapper/ofi_api_wrapper.hpp"

namespace ccl {

lib_info_t ofi_lib_info;
ofi_lib_ops_t ofi_lib_ops;

std::string get_ofi_lib_path() {
    // lib_path specifies the name and full path to the OFI library -
    // it should be an absolute and validated path pointing to the
    // desired libfabric library

    // the order of searching for libfabric is:
    // * CCL_OFI_LIBRARY_PATH (ofi_lib_path env)
    // * I_MPI_OFI_LIBRARY
    // * I_MPI_ROOT/opt/mpi/libfabric/lib
    // * LD_LIBRARY_PATH

    auto ofi_lib_path = ccl::global_data::env().ofi_lib_path;
    if (!ofi_lib_path.empty()) {
        LOG_DEBUG("OFI lib path (CCL_OFI_LIBRARY_PATH): ", ofi_lib_path);
    }
    else {
        char* mpi_ofi_path = getenv("I_MPI_OFI_LIBRARY");
        if (mpi_ofi_path) {
            ofi_lib_path = std::string(mpi_ofi_path);
            LOG_DEBUG("OFI lib path (I_MPI_OFI_LIBRARY): ", ofi_lib_path);
        }
        else {
            char* mpi_root = getenv("I_MPI_ROOT");
            std::string mpi_root_ofi_lib_path =
                mpi_root == NULL ? std::string() : std::string(mpi_root);
            mpi_root_ofi_lib_path += "/opt/mpi/libfabric/lib/libfabric.so";
            struct stat buffer {};
            if (mpi_root && stat(mpi_root_ofi_lib_path.c_str(), &buffer) == 0) {
                ofi_lib_path = std::move(mpi_root_ofi_lib_path);
                LOG_DEBUG("OFI lib path (MPI_ROOT/opt/mpi/libfabric/lib/): ", ofi_lib_path);
            }
            else {
                ofi_lib_path = "libfabric.so";
                LOG_DEBUG("OFI lib path (LD_LIBRARY_PATH): ", ofi_lib_path);
            }
        }
    }

    return ofi_lib_path;
}

std::string get_relative_ofi_lib_path() {
    Dl_info info;

    if (dladdr((void*)ccl::get_library_version, &info)) {
        char libccl_path[PATH_MAX];

        if (realpath(info.dli_fname, libccl_path) != nullptr) {
            // We have to use `realpath`, so the dirname will work correctly,
            // because if there's any symlink like `..` in the path it will not work
            libccl_path[PATH_MAX - 1] = '\0';

            // Remove `libccl.so` from the path to get directory like `$CCL_ROOT/lib`
            char* libccl_dir = dirname(libccl_path);
            // Remove the `lib` from the path the get just the `CCL_ROOT`
            char* ccl_root_cstr = dirname(libccl_dir);

            auto ccl_root = std::string(ccl_root_cstr);
            setenv("FI_PROVIDER_PATH", (ccl_root + "/lib/libfabric/prov").c_str(), 0);
            return ccl_root + "/lib/libfabric/libfabric.so";
        }
    }

    LOG_DEBUG("Could not fetch relative path to libfabric. Fallback to `libfabric.so`");
    return "libfabric.so";
}

bool ofi_api_init() {
    ofi_lib_info.ops = &ofi_lib_ops;
    ofi_lib_info.fn_names = ofi_fn_names;
    ofi_lib_info.path = get_ofi_lib_path();

    int error = load_library(ofi_lib_info);
    if (error == CCL_LOAD_LB_SUCCESS) {
        return true;
    }

    print_error(error, ofi_lib_info);
    LOG_INFO("Retrying to load libfabric.so using relative path");

    ofi_lib_info.path = get_relative_ofi_lib_path();
    error = load_library(ofi_lib_info);
    if (error == CCL_LOAD_LB_SUCCESS) {
        return true;
    }

    print_error(error, ofi_lib_info);
    return false;
}

void ofi_api_fini() {
    LOG_DEBUG("close OFI lib: handle: ", ofi_lib_info.handle);
    close_library(ofi_lib_info);
}

} //namespace ccl
