#
# Copyright 2016-2020 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
file(GLOB sources "*.c" "*.cpp")

link_directories(${EXAMPLES_LIB_DIRS})

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    target_include_directories(${executable} PRIVATE ${EXAMPLES_INC_DIRS})
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC -L${I_MPI_ROOT}/lib/release/)
    target_link_libraries(${executable} PUBLIC mpi)
    target_link_libraries(${executable} PUBLIC ${COMPUTE_BACKEND_TARGET_NAME})
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_EXAMPLES}/external_launcher OPTIONAL)
endforeach()

configure_file(run.sh ${CMAKE_CURRENT_BINARY_DIR}/run.sh @ONLY)
configure_file(run_binary.sh ${CMAKE_CURRENT_BINARY_DIR}/run_binary.sh @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/run.sh DESTINATION ${CCL_INSTALL_EXAMPLES}/external_launcher/  OPTIONAL)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/run_binary.sh DESTINATION ${CCL_INSTALL_EXAMPLES}/external_launcher/ OPTIONAL)
    
