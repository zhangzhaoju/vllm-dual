# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26.1)
project(vllm_ascend_unified LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(PYBIND11_FINDPYTHON ON)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT DEFINED GLIBCXX_USE_CXX11_ABI)
  message(FATAL_ERROR "Pass the ABI of the installed torch; do not guess it")
endif()
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})
add_subdirectory(ascend)
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
    endif()

    if (MARLIN_MOE_FP8_ARCHS)
      file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_FP8_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
      if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
        set_source_files_properties(${MARLIN_MOE_FP8_SRC}
          PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
      endif()
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
    endif()

    set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_OTHER_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

    message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

  # DeepSeek V3 router GEMM kernel - requires SM90+
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0f;11.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(DSV3_ROUTER_GEMM_ARCHS "9.0a;10.0a;10.1a;10.3a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND DSV3_ROUTER_GEMM_ARCHS)
    set(DSV3_ROUTER_GEMM_SRC
      "csrc/moe/dsv3_router_gemm_entry.cu"
      "csrc/moe/dsv3_router_gemm_float_out.cu"
      "csrc/moe/dsv3_router_gemm_bf16_out.cu")
    set_gencode_flags_for_srcs(
      SRCS "${DSV3_ROUTER_GEMM_SRC}"
      CUDA_ARCHS "${DSV3_ROUTER_GEMM_ARCHS}")
    list(APPEND VLLM_MOE_EXT_SRC "${DSV3_ROUTER_GEMM_SRC}")
    message(STATUS "Building DSV3 router GEMM kernel for archs: ${DSV3_ROUTER_GEMM_ARCHS}")
  else()
    message(STATUS "Not building DSV3 router GEMM kernel as no compatible archs found"
                   " (requires SM90+ and CUDA >= 12.0)")
  endif()
endif()

message(STATUS "Enabling moe extension.")
define_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

if(VLLM_GPU_LANG STREQUAL "HIP")
  #
  # _rocm_C extension
  #
  set(VLLM_ROCM_EXT_SRC
    "csrc/rocm/torch_bindings.cpp"
    "csrc/rocm/skinny_gemms.cu"
    "csrc/rocm/attention.cu")

  define_extension_target(
    _rocm_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${VLLM_ROCM_EXT_SRC}
    COMPILE_FLAGS ${VLLM_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    USE_SABI 3
    WITH_SOABI)
endif()

# For CUDA and HIP builds also build the triton_kernels external package.
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
    include(cmake/external_projects/triton_kernels.cmake)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/qutlass.cmake)

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
