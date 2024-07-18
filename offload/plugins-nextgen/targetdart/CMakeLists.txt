##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for targetDART scheduling if available.
#
##===----------------------------------------------------------------------===##

# Looking for MPI...
find_package(MPI QUIET)

if(NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  message("Not building targetDART scheduling plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
endif()

message("Building targetDART NextGen scheduling plugin.")
message(WARNING "Building targetDART NextGen scheduling plugin requires MPI Session support.")

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.targetdart TARGETDART)

target_sources(omptarget.rtl.targetdart PRIVATE
  src/rtl.cpp
)
target_sources(omptarget.rtl.targetdart PRIVATE
  support_structures/src/queue.cpp
)
target_sources(omptarget.rtl.targetdart PRIVATE
  support_structures/src/task.cpp
)
target_sources(omptarget.rtl.targetdart PRIVATE
  support_structures/src/scheduling.cpp
)
target_sources(omptarget.rtl.targetdart PRIVATE
  support_structures/src/communication.cpp
)
target_sources(omptarget.rtl.targetdart PRIVATE
  support_structures/src/threading.cpp
)

target_link_libraries(omptarget.rtl.targetdart PRIVATE MPI::MPI_CXX)


# Add include directories
target_include_directories(omptarget.rtl.targetdart PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

# Configure testing for the MPI plugin.
# list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.targetDART")

# Report to the parent scope that we are building a plugin for MPI.
# set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)