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
#include "sycl_base.hpp"

using namespace std;
using namespace sycl;

/*
    This test operates with four ranks (0, 1, 2, and 3) and checks their behavior across different scenarios.
    Each scenario assigns ranks to specific communicators based on `color`, with `key` controlling ordering
    within each communicator. `key` values are set as `rank % 3`.

    * - Group Members column lists which ranks are part of the same communicator (new_comm) after the split.
    * - Keys column lists the key values used for ordering within the communicator.

    | Scenario     | Ranks   | Color | Keys       | Group Members       | Expected Value per recv_buf[id] |
    |--------------|---------|-------|------------|---------------------|---------------------------------|
    | Scenario 1   | 0, 1, 2 | 1     | 0, 1, 2    | 0, 1, 2             | 3 (0+1+2)                       |
    |              | 3       | 0     | 0          | 3                   | 3                               |
    |--------------|---------|-------|------------|---------------------|---------------------------------|
    | Scenario 2   | 0-3     | 1     | 0, 1, 2, 0 | 0, 1, 2, 3          | 6 (0+1+2+3)                     |
    |--------------|---------|-------|------------|---------------------|---------------------------------|
    | Scenario 3   | 0-3     | 0     | 0, 1, 2, 0 | 0, 1, 2, 3          | 6 (0+1+2+3)                     |
    |--------------|---------|-------|------------|---------------------|---------------------------------|
    | Scenario 4   | 0, 1    | 2     | 0, 1       | 0, 1                | 1 (0+1)                         |
    |              | 2, 3    | 1     | 0, 1       | 2, 3                | 5 (2+3)                         |
    |--------------|---------|-------|------------|---------------------|---------------------------------|
    | Default      | 0, 2    | 0     | 0, 2       | 0, 2                | 2 (0+2)                         |
    |              | 1, 3    | 1     | 1, 0       | 1, 3                | 4 (1+3)                         |
*/

int main(int argc, char* argv[]) {
    size_t count = 1024;

    int size = 0;
    int rank = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    if (size != 4) {
        if (rank == 0) {
            std::cout << "FAILED: This test requires exactly 4 ranks.\n";
        }
        return -1;
    }

    queue q;
    sycl::property_list props;
    if (argc > 3) {
        if (strcmp("in_order", argv[3]) == 0) {
            props = { sycl::property::queue::in_order{} };
        }
    }

    if (argc > 4) {
        count = (size_t)std::atoi(argv[4]);
    }

    if (!create_sycl_queue(argc, argv, rank, q, props)) {
        return -1;
    }

    buf_allocator<int> allocator(q);

    auto usm_alloc_type = usm::alloc::shared;
    if (argc > 2) {
        usm_alloc_type = usm_alloc_type_from_string(argv[2]);
    }

    if (!check_sycl_usm(q, usm_alloc_type)) {
        return -1;
    }

    int scenario = (argc > 5) ? std::atoi(argv[5]) : 0;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void*)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    auto stream = ccl::create_stream(q);

    auto send_buf = allocator.allocate(count * size, usm_alloc_type); // Adjusted for reduce_scatter
    auto recv_buf = allocator.allocate(count, usm_alloc_type);

    sycl::buffer<int> expected_buf(count);
    int color = 0;
    int key = rank % 3;

    // Assign color based on scenario
    switch (scenario) {
        case 1: color = (rank < 3) ? 1 : 0; break;
        case 2: color = 1; break;
        case 3: color = 0; break;
        case 4: color = (rank < 2) ? 2 : 1; break;
        default: color = (rank % 2 == 0) ? 0 : 1; break;
    }

    auto new_comm = ccl::split_communicator(comm, color, key);

    // Fill send_buf with rank values and initialize recv_buf
    auto e = q.submit([&](auto& h) {
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            recv_buf[id] = -1;
            expected_buf_acc[id] = 0; // Initialize expected buffer to 0
            for (int i = 0; i < size; i++) {
                send_buf[i * count + id] = rank; // Fill send_buf with rank
            }
        });
    });

    vector<ccl::event> deps;
    deps.push_back(ccl::create_event(e));
    sycl::buffer<int> check_buf(count);
    auto attr = ccl::create_operation_attr<ccl::reduce_scatter_attr>();
    ccl::reduce_scatter(send_buf,
                        recv_buf,
                        count,
                        ccl::datatype::int32,
                        ccl::reduction::sum,
                        new_comm,
                        stream,
                        attr,
                        deps)
        .wait();

    // Calculate expected values based on communicator group
    q.submit([&](auto& h) {
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::write_only);
        h.parallel_for(count, [=](auto id) {
            int group_sum = 0;

            switch (scenario) {
                case 1:
                    if (color == 1) { // Group {0, 1, 2}
                        group_sum = 0 + 1 + 2; // Sum of ranks 0, 1, 2
                    }
                    else { // Rank 3 alone
                        group_sum = 3;
                    }
                    break;

                case 2:
                case 3:
                    group_sum = 0 + 1 + 2 + 3; // All ranks together
                    break;

                case 4:
                    if (color == 2) { // Group {0, 1}
                        group_sum = 0 + 1;
                    }
                    else if (color == 1) { // Group {2, 3}
                        group_sum = 2 + 3;
                    }
                    break;

                default:
                    if (color == 0) { // Group {0, 2}
                        group_sum = 0 + 2;
                    }
                    else if (color == 1) { // Group {1, 3}
                        group_sum = 1 + 3;
                    }
                    break;
            }

            expected_buf_acc[id] = group_sum;
        });
    });

    // Validation logic
    q.submit([&](auto& h) {
        sycl::accessor check_buf_acc(check_buf, h, sycl::write_only);
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::read_only);
        h.parallel_for(count, [=](auto id) {
            check_buf_acc[id] = (recv_buf[id] != expected_buf_acc[id]) ? -1 : 0;
        });
    });

    if (!handle_exception(q))
        return -1;

    /* Print result of the test */
    {
        sycl::host_accessor check_buf_acc(check_buf, sycl::read_only);
        size_t i;
        for (i = 0; i < count; i++) {
            if (check_buf_acc[i] == -1) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == count) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}
