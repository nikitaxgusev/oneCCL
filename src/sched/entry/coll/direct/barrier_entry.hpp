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
#pragma once

#include "sched/entry/coll/direct/base_coll_entry.hpp"

class barrier_entry : public base_coll_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "BARRIER";
    }

    barrier_entry() = delete;
    barrier_entry(ccl_sched* sched, ccl_comm* comm) : base_coll_entry(sched), comm(comm) {}

    void start() override {
        LOG_DEBUG("BARRIER entry req ", &req);

        atl_status_t atl_status = comm->get_atl_comm()->barrier(sched->bin->get_atl_ep(), req);
        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("BARRIER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }
        else {
            status = ccl_sched_entry_status_started;
        }
    }

    void update() override {
        atl_status_t atl_status = comm->get_atl_comm()->check(sched->bin->get_atl_ep(), req);

        if (unlikely(atl_status != ATL_STATUS_SUCCESS)) {
            CCL_THROW("BARRIER entry failed. atl_status: ", atl_status_to_str(atl_status));
        }

        if (req.is_completed) {
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "comm_id ", comm->get_comm_id(), ", req ", req, "\n");
    }

private:
    ccl_comm* comm;
    atl_req_t req{};
};
