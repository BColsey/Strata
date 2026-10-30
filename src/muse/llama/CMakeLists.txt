cmake_minimum_required(VERSION 3.24)
project(strata_muse_llama LANGUAGES C CXX)
# Separate project: never changes Strata's GGML pin or its CUDA/HIP/SYCL graph.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The experimental Muse llama adapter currently supports Linux CPU/CUDA builds only")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(GGML_NATIVE OFF CACHE BOOL "Portable CPU build")
set(GGML_CUDA OFF CACHE BOOL "Enable the opt-in CUDA backend")
set(GGML_BLAS OFF CACHE BOOL "No external BLAS dependency")
set(GGML_CCACHE OFF CACHE BOOL "No implicit compiler cache")
foreach(option IN ITEMS LLAMA_BUILD_COMMON LLAMA_BUILD_EXAMPLES LLAMA_BUILD_TOOLS
    LLAMA_BUILD_TESTS LLAMA_BUILD_SERVER LLAMA_BUILD_APP LLAMA_BUILD_UI LLAMA_OPENSSL)
  set(${option} OFF CACHE BOOL "Not needed by the token-ID adapter" FORCE)
endforeach()
include(FetchContent)
set(STRATA_MUSE_LLAMA_ARCHIVE
  "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/e09a78a25dec3f42175ad59ecc540a54bed0b9b2"
  CACHE STRING "Pinned source archive URL or local file (same SHA256 required)")
FetchContent_Declare(muse_llama URL "${STRATA_MUSE_LLAMA_ARCHIVE}"
  URL_HASH SHA256=398e341ee555ba87d829b3ee9482048d62dd80859bec81e2459325ee18211dc5
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(muse_llama)
find_package(Threads REQUIRED)
get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(strata-muse-llama main.cpp ../runtime.cpp ../protocol.cpp)
target_include_directories(strata-muse-llama PRIVATE "${STRATA_ROOT}/include")
target_link_libraries(strata-muse-llama PRIVATE llama ggml Threads::Threads)
target_compile_options(strata-muse-llama PRIVATE -Wall -Wextra)
set_target_properties(strata-muse-llama PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE INSTALL_RPATH "$ORIGIN")
include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME muse_llama_protocol COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../test_protocol.py" --binary $<TARGET_FILE:strata-muse-llama>)
  set_tests_properties(muse_llama_protocol PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
  add_test(NAME muse_llama_logits COMMAND "${Python3_EXECUTABLE}" -m unittest tools.muse.test_llama_backend -v)
  set_tests_properties(muse_llama_logits PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "${STRATA_ROOT}"
    ENVIRONMENT "STRATA_MUSE_LLAMA_TEST_EXE=$<TARGET_FILE:strata-muse-llama>")
endif()
