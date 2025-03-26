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
#include "coll/coll_util.hpp"
#include "coll/group/group.hpp"
#include "common/global/global.hpp"

thread_local bool group_impl::is_group_active = false;
thread_local bool group_impl::first_group_op = false;
thread_local std::vector<std::pair<ccl_coll_type, std::function<ccl::event()>>>
    group_impl::operation_storage;
std::mutex group_impl::group_mutex;

void group_impl::start() {
    std::lock_guard<std::mutex> lock(group_mutex);
    LOG_INFO("group operation is started");
    operation_storage.clear();
    is_group_active = true;
    ccl::enable_direct_fallback_for_pt2pt();
}

void group_impl::end() {
    std::lock_guard<std::mutex> lock(group_mutex);
    if (is_group_active) {
#ifdef CCL_ENABLE_SYCL
        auto store_ze_pt2pt_read = ccl::global_data::env().ze_pt2pt_read;
        // currently for group API only ze_pt2pt_read = 1 is supported
        ccl::global_data::env().ze_pt2pt_read = 1;
#endif // CCL_ENABLE_SYCL
        first_group_op = true;
        ccl::event event;
        for (const auto& operation : operation_storage) {
            event = operation.second();
            first_group_op = false;
        }
        first_group_op = false; // needed in case operation_storage is empty
        // wait() is needed to avoid oneCCL destruction prior to device tasks completion
        event.wait();
#ifdef CCL_ENABLE_SYCL
        ccl::global_data::env().ze_pt2pt_read = store_ze_pt2pt_read;
#endif // CCL_ENABLE_SYCL
    }
    ccl::restore_pt2pt_fallback_table();
    LOG_INFO("group operation is ended");
    is_group_active = false;
    operation_storage.clear();
}

void group_impl::add_operation(ccl_coll_type ctype, std::function<ccl::event()> operation) {
    if (is_group_active) {
        operation_storage.push_back(std::make_pair(ctype, std::move(operation)));
    }
    else {
        CCL_THROW("group_impl is not actived");
    }
}
