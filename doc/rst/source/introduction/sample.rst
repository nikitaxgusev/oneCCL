==========================
Build a Sample Application
==========================

The sample code below shows how to use |product_short| API to perform allreduce communication for SYCL USM memory.

.. literalinclude:: sample.cpp
   :language: cpp


Build the Library
*****************

#. Build the library with ``SYCL`` support (only Intel\ |reg|\  oneAPI DPC++/C++ Compiler is supported).

   ::

     cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCOMPUTE_BACKEND=dpcpp
     make install

#. `Set up the library environment <https://www.intel.com/content/www/us/en/docs/oneccl/get-started-guide/2021-13/overview.html#BEFORE-YOU-BEGIN>`_.

#. Use the C++ driver with the -fsycl option to build the sample:

   ::

      icpx -o sample sample.cpp -lccl -lmpi -fsycl


Run the Sample
**************

Intel\ |reg|\  MPI Library is required for running the sample. Make sure that MPI environment is set up.

To run the sample, use the following command:

::

    mpiexec <parameters> ./sample

where ``<parameters>`` represents optional mpiexec parameters such as node count, processes per node, hosts, and so on.

.. note:: Explore the complete list of oneAPI code samples in the `oneAPI Samples Catalog <https://oneapi-src.github.io/oneAPI-samples/>`_. These samples were designed to help you develop, offload, and optimize multiarchitecture applications targeting CPUs, GPUs, and FPGAs.
