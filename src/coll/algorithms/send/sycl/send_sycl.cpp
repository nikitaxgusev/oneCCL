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
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/algorithms/utils/sycl_coll_base.hpp"
#include "common/request/request.hpp"
#include "common/event/impls/host_event.hpp"
#include "coll/algorithms/send/sycl/send_sycl.hpp"
#include "sched/entry/ze/ze_pt2pt_barrier_entry.hpp"

namespace ccl {
namespace v1 {

static ccl::event send_sycl_single_node(sycl::queue& q,
                                        const void* send_buf,
                                        size_t send_count,
                                        ccl::datatype dtype,
                                        int peer_rank,
                                        ccl_comm* comm,
                                        ccl_stream* global_stream,
                                        const pt2pt_attr& attr,
                                        const std::vector<ccl::event>& deps,
                                        bool& done) {
    ccl_comm* node_comm = comm->get_node_comm().get();
    auto node_peer_rank = node_comm->get_rank_from_global(peer_rank);
    auto node_curr_rank = node_comm->rank();
    auto sycl_queue = global_stream->get_native_stream();

    LOG_DEBUG("send_sycl_single_node: send_buf=",
              send_buf,
              ", count=",
              send_count,
              ", peer_rank=",
              node_peer_rank);

    auto* tagc = node_comm->get_atl_comm()->tag_creator.get();
    auto comm_id = comm->get_atl_comm()->get_comm_id();
    auto [sync_ready, sync_done] = tagc->get_pt2pt_sync_tags();
    uint64_t tag_ready = tagc->create(node_peer_rank, comm_id, sync_ready);
    uint64_t tag_done = tagc->create(node_peer_rank, comm_id, sync_done);

    if (send_count == 0 || comm->size() == 1) {
        LOG_DEBUG("send_sycl_single_node: count is 0 or comm size is 1, skipping send");
        auto sycl_deps = get_sycl_events(deps);
        sycl::event barrier = submit_wait_on_events(sycl_queue, sycl_deps);

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ barrier },
                                                   comm,
                                                   /*do_send_ack*/ false,
                                                   node_peer_rank,
                                                   tag_done);
        done = true;
        return ccl::event::create_from_native(ack_event);
    }

    // for ARC GPUs to do ring LL256
    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device())) &&
        !group_impl::is_group_active) {
        ccl::event e =
            send_ll(send_buf, send_count, dtype, peer_rank, comm, global_stream, deps, done);
        return e;
    }

    std::vector<ze_handle_exchange_entry::mem_desc_t> buffer{ { const_cast<void*>(send_buf),
                                                                ccl::ze::ipc_mem_type::memory } };

    ccl::utils::pt2pt_handle_exchange_info info{ node_peer_rank,
                                                 ccl::utils::pt2pt_handle_exchange_role::sender };
    if (!ccl::global_data::env().sycl_pt2pt_read) {
        info.role = ccl::utils::pt2pt_handle_exchange_role::receiver;
    }

    ccl_coll_param param = ccl_coll_param::create_send_param(
        send_buf, send_count, dtype, peer_rank, attr, comm, global_stream, deps);
    ccl_coll_attr coll_attr{};
    ccl_sched* sched = ccl_sched::create(param, coll_attr);

    auto* exchange_entry =
        new ze_handle_exchange_entry(sched, node_comm, buffer, ccl_comm::invalid_rank, info);
    exchange_entry->start();

    while (!exchange_entry->is_completed()) {
        exchange_entry->update();
    }

    ccl::event ret_evt;
    auto sycl_deps = get_sycl_events(deps);

    // Write mode => we do a device memcpy (local->remote) then ack-report
    if (!ccl::global_data::env().sycl_pt2pt_read) {
        LOG_DEBUG("send_sycl_single_node: write mode enabled");

        // in write mode we have to wait for confirmation from receiver to ensure readiness
        sycl::event sync_event = pt2pt_pre_sync(sycl_queue,
                                                sycl_deps,
                                                comm,
                                                /*do_send=*/false,
                                                node_peer_rank,
                                                tag_ready);

        ccl_buffer out_buf;
        sched->get_memory().handle_manager.get(0, 0, out_buf, node_comm, /*pt2pt_op=*/true);

        CCL_THROW_IF_NOT(out_buf.get_ptr(), "no pointer from peer in write mode");

        int bytes = ccl::get_datatype_size(dtype) * send_count;
        sycl::event copy_event = sycl_queue.memcpy(
            out_buf.get_ptr(), send_buf, bytes, std::vector<sycl::event>{ sync_event });

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ copy_event },
                                                   comm,
                                                   /*do_send=*/true,
                                                   node_peer_rank,
                                                   tag_done);

        ret_evt = ccl::event::create_from_native(ack_event);

        LOG_DEBUG("send_sycl_single_node: ack_report done, ack_tag=", tag_done);
    }
    // Read mode => remote side will do the copy, we only do ack-accept
    else {
        LOG_DEBUG("send_sycl_single_node: read mode enabled");

        // in read mode we have to send sync confirmation to reader (reciever)
        // this is needed to avoid situation when receiver will read before the data is ready
        sycl::event sync_event = pt2pt_pre_sync(sycl_queue,
                                                sycl_deps,
                                                comm,
                                                /*do_send=*/true,
                                                node_peer_rank,
                                                tag_ready);

        sycl::event ack_event = post_host_task_ack(sycl_queue,
                                                   std::vector<sycl::event>{ sync_event },
                                                   comm,
                                                   /*do_send=*/false,
                                                   node_peer_rank,
                                                   tag_done);

        ret_evt = ccl::event::create_from_native(ack_event);

        LOG_DEBUG("send_sycl_single_node: ack_accept done, ack_tag=", tag_done);
    }

    done = true;
    return ret_evt;
}

ccl::event send_sycl(sycl::queue& q,
                     const void* send_buf,
                     size_t send_count,
                     ccl::datatype dtype,
                     int peer_rank,
                     ccl_comm* comm,
                     ccl_stream* global_stream,
                     const pt2pt_attr& attr,
                     const vector_class<event>& deps,
                     bool& done) {
    bool is_single_node = false;
    bool is_single_card = false;
    if (ccl::global_data::env().backend == backend_mode::native) {
        const ccl::topo_manager& topo_manager = comm->get_topo_manager();
        is_single_node = topo_manager.is_single_node;
        is_single_card = topo_manager.is_single_card;
    }

    if (is_single_card) {
        LOG_DEBUG("send_sycl: read mode enabled: tiles on the same card are detected");
        ccl::global_data::env().sycl_pt2pt_read = 1;
    }

    if (is_arc_card(ccl::ze::get_device_family(global_stream->get_ze_device()))) {
        LOG_DEBUG("recv_sycl: write mode enabled: ARC card detected");
        ccl::global_data::env().sycl_pt2pt_read = 0;
    }

    if (is_single_node) {
        return send_sycl_single_node(
            q, send_buf, send_count, dtype, peer_rank, comm, global_stream, attr, deps, done);
    }
    else {
        done = false;
        CCL_THROW("send_sycl: multi-node case is not supported yet");
        return ccl::event();
    }
}

} // namespace v1
} // namespace ccl
