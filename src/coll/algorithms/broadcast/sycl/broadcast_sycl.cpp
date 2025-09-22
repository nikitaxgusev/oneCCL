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
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "coll/algorithms/broadcast/sycl/broadcast_sycl.hpp"

namespace ccl {
namespace v1 {

ccl::event broadcast_sycl_single_node(sycl::queue& q,
                                      const void* send_buf,
                                      void* recv_buf,
                                      size_t count,
                                      ccl::datatype dtype,
                                      int root,
                                      ccl_comm* comm,
                                      ccl_stream* global_stream,
                                      const vector_class<event>& deps,
                                      bool& done) {
    done = true;
    auto ccl_dtype = ccl::global_data::get().dtypes->get(dtype);

#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_begin("broadcast_small", "send_size", count * ccl_dtype.size());
#endif // CCL_ENABLE_ITT
    LOG_DEBUG("|CCL_SYCL| broadcast selects small kernel, count: ", count, " datatype: ", dtype);
    ccl::event e =
        broadcast_small(send_buf, recv_buf, count, dtype, root, comm, global_stream, deps);
    LOG_DEBUG(
        "|CCL_SYCL| broadcast selects small kernel, count: ", count, " datatype: ", dtype, " done");
#ifdef CCL_ENABLE_ITT
    ccl::profile::itt::task_end();
#endif // CCL_ENABLE_ITT

    return e;
}

} // namespace v1
} // namespace ccl
