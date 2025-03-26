.. _mpi: https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html

============
Introduction
============

|product_full| (|product_short|) provides an efficient implementation of communication patterns used in deep learning. 

|product_short| features include:

- Built on top of lower-level communication middleware – |mpi|_ and `libfabrics <https://github.com/ofiwg/libfabric>`_.
- Optimized to drive scalability of communication patterns. It supports collectives and point-to-point (send/receive) primitives.
- Works across various interconnects: InfiniBand*, Cornelis Networks*, and Ethernet.
- Provides common API sufficient to support communication workflows within Deep Learning / distributed frameworks (such as `PyTorch* <https://github.com/pytorch/pytorch>`_, `Horovod* <https://github.com/horovod/horovod>`_).

|product_short| package comprises the |product_short| Software Development Kit (SDK) and the |mpi| Runtime components.