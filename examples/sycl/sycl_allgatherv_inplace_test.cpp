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

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage ./sycl_allgatherv_inplace_test [device]\n";
        cout << "device could be 'cpu' or 'gpu'\n";
        cout << "example: ./sycl_allgatherv_inplace_test cpu\n";
        exit(1);
    }

    string device_type = argv[1];

    if (device_type != "cpu" && device_type != "gpu") {
        cout << "error: Invalid device '" << device_type << "'.\n";
        cout << "device must be either 'cpu' or 'gpu'.\n";
        exit(1);
    }

    size_t count = 10 * 1024 * 1024;

    int size = 0;
    int rank = 0;
    size_t send_buf_count = 0;
    size_t recv_buf_count = 0;

    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    test_args args(argc, argv, rank);

    if (args.count != args.DEFAULT_COUNT) {
        count = args.count;
    }

    sycl::queue q;
    if (!create_test_sycl_queue(device_type, rank, q, args))
        return -1;

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* create buffers */
    send_buf_count = count + rank;
    recv_buf_count = count * size + ((size - 1) * size) / 2;

    sycl::buffer<int> send_buf(send_buf_count);
    sycl::buffer<int> expected_buf(recv_buf_count);
    sycl::buffer<int> recv_buf(recv_buf_count);

    std::vector<size_t> recv_counts(size, 0);
    iota(recv_counts.begin(), recv_counts.end(), count);

    {
        /* open buffers and initialize them on the host side */
        sycl::host_accessor send_buf_acc(send_buf, sycl::write_only);
        sycl::host_accessor recv_buf_acc(recv_buf, sycl::write_only);
        sycl::host_accessor expected_acc_buf(expected_buf, sycl::write_only);

        for (size_t i = 0; i < send_buf_count; i++) {
            send_buf_acc[i] = rank;
        }
        for (size_t i = 0; i < recv_buf_count; i++) {
            recv_buf_acc[i] = -1;
        }
        size_t idx = 0;
        for (int i = 0; i < size; i++) {
            for (size_t j = 0; j < count + i; j++) {
                expected_acc_buf[idx + j] = i + 1;
            }
            idx += count + i;
        }
    }

    /* open send_buf and modify it on the device side */
    /* make in-place updates in the appropriate place */
    size_t rbuf_idx = 0;
    for (int i = 0; i < rank; i++)
        rbuf_idx += recv_counts[i];

    q.submit([&](auto &h) {
        sycl::accessor send_buf_acc(send_buf, h, sycl::read_only);
        sycl::accessor recv_buf_acc(recv_buf, h, sycl::write_only);
        h.parallel_for(send_buf_count, [=](auto id) {
            recv_buf_acc[rbuf_idx + id] = send_buf_acc[id] + 1;
        });
    });

    if (!handle_exception(q))
        return -1;

    /* invoke allgatherv */
    /* make in-place operation by providing recv_buf as input */
    buffer recv_buff_with_offset(recv_buf, sycl::id{ rbuf_idx }, sycl::range{ recv_counts[rank] });

    ccl::allgatherv(recv_buff_with_offset, send_buf_count, recv_buf, recv_counts, comm, stream)
        .wait();

    /* open recv_buf and check its correctness on the device side */
    q.submit([&](auto &h) {
        sycl::accessor recv_buf_acc(recv_buf, h, sycl::write_only);
        sycl::accessor expected_buf_acc(expected_buf, h, sycl::read_only);
        h.parallel_for(recv_buf_count, [=](auto id) {
            if (recv_buf_acc[id] != expected_buf_acc[id]) {
                recv_buf_acc[id] = -1;
            }
        });
    });

    if (!handle_exception(q))
        return -1;

    /* print out the result of the test on the host side */
    {
        sycl::host_accessor recv_buf_acc(recv_buf, sycl::read_only);
        size_t i;
        for (i = 0; i < recv_buf_count; i++) {
            if (recv_buf_acc[i] == -1) {
                std::cout << "FAILED\n";
                break;
            }
        }
        if (i == recv_buf_count) {
            std::cout << "PASSED\n";
        }
    }

    return 0;
}
