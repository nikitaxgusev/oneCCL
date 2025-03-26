.. _`oneCCL specification`: https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneCCL/source/index.html
Programming Model
=================

The programming model for oneCCL describes how to:

* Set up distributed computations.

* Perform collective communication operations (for example, ALLREDUCE, BROADCAST, ALLGATHER).

.. seealso:: See `oneCCL specification`_ that oneCCL is based on.


oneCCL supports a single rank/process per GPU device. The current implementation does not yet support a single process opening multiple devices.


Review the oneCCL `generic workflow <https://uxlfoundation.github.io/oneAPI-spec/spec/elements/oneCCL/source/spec/generic_workflow.html>`_ in the specification before getting started with the communication operations.

You can quickly get started with:

.. toctree::
   :maxdepth: 1

   programming-model/host-communication.rst
   programming-model/device-communication.rst
   


