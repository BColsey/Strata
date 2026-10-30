#pragma once
#include "strata/muse/model.hpp"
#include <string_view>

namespace strata::muse {
constexpr size_t resident_line_bytes = 16 * 1024 * 1024;
// Adds resident command/history storage to the unchanged numerical CLI budget.
size_t resident_memory_bytes(const Geometry& geometry, size_t context);
std::string protocol_error(std::string_view message);
} // namespace strata::muse
