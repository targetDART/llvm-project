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
  libomptarget_say("Not building MPI offloading plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
elseif(NOT MPI_CXX_FOUND)
  libomptarget_say("Not building MPI offloading plugin: MPI not found in system.")
  return()
endif()

libomptarget_say("Building targetDART NextGen scheduling plugin.")

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.targetDART TARGETDART)

target_sources(omptarget.rtl.targetDART PRIVATE
  src/rtl.cpp
)

target_link_libraries(omptarget.rtl.targetDART PRIVATE
  MPI::MPI_CXX
)

# Add include directories
target_include_directories(omptarget.rtl.targetDART PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

# Configure testing for the MPI plugin.
# list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.targetDART")

# Report to the parent scope that we are building a plugin for MPI.
# set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)