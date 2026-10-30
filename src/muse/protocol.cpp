#include "protocol.hpp"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <locale>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace strata::muse {
size_t resident_memory_bytes(const Geometry& geometry, size_t context) {
    const auto bytes = checked_add(geometry.memory_bytes(context),
        checked_add(resident_line_bytes, checked_mul(context, size_t(16))));
    if (bytes > host_budget_bytes)
        throw std::runtime_error("Muse: KV, scratch and resident protocol storage exceed 1 GiB; reduce context");
    return bytes;
}
std::string protocol_error(std::string_view message) {
    std::string clean(message.substr(0, 1024));
    std::replace_if(clean.begin(), clean.end(), [](char c) { return c == '\r' || c == '\n' || c == '\0'; }, ' ');
    return "ERR " + clean;
}
namespace {
using Clock = std::chrono::steady_clock;
struct Cancelled {};
[[noreturn]] void bad(const std::string& message) { throw std::runtime_error("Muse: " + message); }

// getline would allocate without a bound before we could reject a hostile line.
// Discard its remainder so a later valid command can still be read.
bool read_line(char* line, size_t& size, bool& oversized) {
    size = 0; oversized = false;
    char c;
    bool read = false;
    while (std::cin.get(c)) {
        read = true;
        if (c == '\n') break;
        if (size < resident_line_bytes) line[size++] = c;
        else oversized = true;
    }
    if (size && line[size - 1] == '\r') --size;
    return read;
}
std::string_view word(std::string_view& text) {
    const auto first = text.find_first_not_of(" \t");
    if (first == text.npos) { text = {}; return {}; }
    text.remove_prefix(first);
    const auto end = text.find_first_of(" \t");
    const auto out = text.substr(0, end);
    text = end == text.npos ? std::string_view{} : text.substr(end);
    return out;
}
uint64_t integer(std::string_view text, const char* name) {
    if (text.empty()) bad(std::string("missing ") + name);
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size())
        bad(std::string("invalid/out-of-range ") + name);
    return value;
}
double real(std::string_view text, const char* name) {
    // Bound inputs to finite float range, while doing penalty/sampling arithmetic
    // in double so small positive temperatures and repeat penalties stay finite.
    if (text.empty()) bad(std::string("missing ") + name);
    float value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(value))
        bad(std::string("invalid/nonfinite ") + name);
    return value;
}
struct Request {
    uint32_t maximum = 0, top_k = 64, last_n = 64;
    double temperature = 0, top_p = 1, min_p = 0, repeat = 1, frequency = 0, presence = 0;
    uint64_t seed = 0;
    std::vector<uint32_t> tokens;
};
Request parse_request(std::string_view text, size_t context, size_t vocab) {
    Request r;
    const auto maximum = integer(word(text), "maximum tokens");
    if (maximum > context) bad("generation exceeds context");
    r.maximum = uint32_t(maximum);
    std::set<std::string> seen;
    std::string_view ids;
    while (true) {
        const auto token = word(text);
        if (token.empty()) break;
        const auto eq = token.find('=');
        if (eq == token.npos) { ids = token; break; }
        if (eq > 32) bad("invalid sampling option");
        const std::string key(token.substr(0, eq));
        if (!seen.insert(key).second) bad("duplicate sampling option");
        const auto value = token.substr(eq + 1);
        if (key == "seed") r.seed = integer(value, "seed");
        else if (key == "top_k") {
            const auto n = integer(value, "top_k");
            r.top_k = n == 0 || n > 64 ? 64 : uint32_t(n);
        } else if (key == "penalty_last_n") {
            const auto n = integer(value, "penalty_last_n");
            if (n > UINT32_MAX) bad("penalty_last_n exceeds uint32 range");
            r.last_n = uint32_t(n);
        } else if (key == "temperature") r.temperature = real(value, "temperature");
        else if (key == "top_p") r.top_p = real(value, "top_p");
        else if (key == "min_p") r.min_p = real(value, "min_p");
        else if (key == "penalty_repeat") r.repeat = real(value, "penalty_repeat");
        else if (key == "penalty_freq") r.frequency = real(value, "penalty_freq");
        else if (key == "penalty_present") r.presence = real(value, "penalty_present");
        else bad("unsupported sampling option: " + key);
    }
    if (r.temperature < 0 || r.top_p <= 0 || r.top_p > 1 || r.min_p < 0 || r.min_p > 1 || r.repeat <= 0)
        bad("sampling option outside its valid range");
    if (ids.empty() || !word(text).empty()) bad("expected one comma-separated prompt token list");
    r.tokens.reserve(context - r.maximum);
    while (true) {
        if (r.tokens.size() >= context - r.maximum) bad("prompt plus generation exceeds context");
        const auto comma = ids.find(',');
        const auto id = integer(ids.substr(0, comma), "token");
        if (id >= vocab) bad("token outside vocabulary");
        r.tokens.push_back(uint32_t(id));
        if (comma == ids.npos) break;
        ids.remove_prefix(comma + 1);
    }
    return r;
}

uint32_t sample(const std::vector<float>& logits, const std::vector<uint32_t>& history,
                const Request& r, std::mt19937_64& rng) {
    std::vector<uint32_t> count(logits.size());
    const size_t first = history.size() - std::min(history.size(), size_t(r.last_n));
    for (size_t i = first; i < history.size(); ++i) ++count[history[i]];
    std::vector<double> adjusted(logits.begin(), logits.end());
    for (size_t i = 0; i < adjusted.size(); ++i) if (count[i]) {
        adjusted[i] = adjusted[i] <= 0 ? adjusted[i] * r.repeat : adjusted[i] / r.repeat;
        adjusted[i] -= count[i] * r.frequency + r.presence;
    }
    const auto best = std::max_element(adjusted.begin(), adjusted.end());
    if (r.temperature == 0) return uint32_t(best - adjusted.begin());
    std::vector<uint32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0u);
    size_t keep = std::min(size_t(r.top_k), ids.size());
    std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(), [&](uint32_t a, uint32_t b) {
        return adjusted[a] == adjusted[b] ? a < b : adjusted[a] > adjusted[b];
    });
    // Match the normal engine's order: truncate raw-logit top_p/min_p before
    // temperature. Muse's seeded RNG is independent of the Qwen GPU sampler.
    const double maximum = adjusted[ids[0]];
    if (r.top_p < 1) {
        double sum = 0, cumulative = 0;
        for (size_t i = 0; i < keep; ++i) sum += std::exp(adjusted[ids[i]] - maximum);
        for (size_t i = 0; i < keep; ++i) {
            cumulative += std::exp(adjusted[ids[i]] - maximum);
            if (cumulative >= r.top_p * sum) { keep = i + 1; break; }
        }
    }
    if (r.min_p > 0) {
        const auto threshold = maximum + std::log(r.min_p);
        for (size_t i = 1; i < keep; ++i) if (adjusted[ids[i]] < threshold) { keep = i; break; }
    }
    double sum = 0;
    for (size_t i = 0; i < keep; ++i) sum += std::exp((adjusted[ids[i]] - maximum) / r.temperature);
    double draw = double(rng() >> 11) * (1.0 / 9007199254740992.0) * sum;
    for (size_t i = 0; i < keep; ++i) {
        draw -= std::exp((adjusted[ids[i]] - maximum) / r.temperature);
        if (draw < 0) return ids[i];
    }
    return ids[keep - 1];
}
double milliseconds(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}
class Resident {
    Execution& model;
    const size_t context;
    const std::string backend;
    std::mutex state_mutex, output_mutex;
    std::condition_variable changed;
    std::optional<Request> pending;
    bool active = false, quitting = false;
    std::atomic<bool> cancelled{false};
    std::thread worker;

    void line(const std::string& text) {
        std::lock_guard<std::mutex> lock(output_mutex);
        std::cout << text << '\n' << std::flush;
    }
    void error(const std::string& message) {
        line(protocol_error(message));
    }
    std::string generate(const Request& request) {
        const auto start = Clock::now();
        auto prompt_end = start, heartbeat = start;
        size_t processed = 0, generated = 0;
        bool decoding = false;
        std::string finish = "length";
        auto progress = [&] {
            heartbeat = Clock::now();
            line("PP " + std::to_string(processed) + " " + std::to_string(request.tokens.size()));
        };
        const std::function<void()> checkpoint = [&] {
            if (cancelled.load(std::memory_order_relaxed)) throw Cancelled{};
            if (Clock::now() - heartbeat >= std::chrono::seconds(1)) progress();
        };
        try {
            model.reset();
            progress();
            std::vector<float> logits;
            for (auto token : request.tokens) {
                logits = model.forward(token, checkpoint);
                ++processed;
                if (processed == request.tokens.size() || processed % 16 == 0) progress();
            }
            prompt_end = Clock::now(); decoding = true;
            std::vector<uint32_t> history;
            history.reserve(context);
            history.insert(history.end(), request.tokens.begin(), request.tokens.end());
            uint64_t seed = request.seed;
            if (!seed) { std::random_device device; seed = (uint64_t(device()) << 32) ^ device(); }
            std::mt19937_64 rng(seed);
            for (; generated < request.maximum;) {
                checkpoint();
                const auto token = sample(logits, history, request, rng);
                checkpoint();
                line("T " + std::to_string(token));
                ++generated;
                history.push_back(token);
                if (model.is_end(token)) { finish = "stop"; break; }
                if (generated < request.maximum) logits = model.forward(token, checkpoint);
            }
        } catch (const Cancelled&) {
            finish = "cancel";
            if (!decoding) prompt_end = Clock::now();
        }
        const auto end = Clock::now();
        std::ostringstream done;
        done.imbue(std::locale::classic());
        done << "DONE " << generated << ' ' << request.tokens.size() << ' '
             << milliseconds(start, prompt_end) << ' ' << milliseconds(prompt_end, end) << ' ' << finish;
        return done.str();
    }
    void work() {
        while (true) {
            std::unique_lock<std::mutex> lock(state_mutex);
            changed.wait(lock, [&] { return quitting || pending.has_value(); });
            if (!pending) return;
            auto request = std::move(*pending); pending.reset();
            lock.unlock();
            std::string result;
            try { result = generate(request); }
            catch (const std::exception& e) { result = protocol_error(e.what()); }
            catch (...) { result = protocol_error("Muse: request failed"); }
            lock.lock();
            // Publish completion before admitting the next command. A STOP that
            // races with completion can never cancel the following request.
            active = false;
            line(result);
        }
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            quitting = true;
            cancelled.store(true, std::memory_order_relaxed);
        }
        changed.notify_one();
        if (worker.joinable()) worker.join();
    }
public:
    Resident(const Options& options, Execution& execution)
        : model(execution), context(options.context), backend(options.backend) {
        if (backend == "cpu" || backend == "cuda") resident_memory_bytes(model.geometry(), context);
        else if (checked_add(resident_line_bytes, checked_mul(context, size_t(16))) > host_budget_bytes)
            bad("resident protocol buffers exceed 1 GiB; reduce context");
    }
    ~Resident() { stop(); }
    void run() {
        // Fixed storage avoids std::string growth capacity exceeding the line
        // allowance accounted by resident_memory_bytes().
        auto input = std::make_unique<char[]>(resident_line_bytes);
        worker = std::thread([&] { work(); });
        const std::string engine = backend == "cpu" || backend == "cuda" ? "muse-reference" : "muse-llama";
        line("INFO arch=muse-spark backend=" + backend + " engine=" + engine + " batch_slots=0 logprobs=0 prefix_cache=0");
        line("READY " + std::to_string(context) + " stop");
        size_t input_size = 0;
        bool oversized = false;
        while (read_line(input.get(), input_size, oversized)) {
            if (oversized) { error("Muse: command exceeds 16 MiB limit"); continue; }
            try {
                std::string_view rest(input.get(), input_size);
                const auto command = word(rest);
                if (command == "STOP") {
                    if (!word(rest).empty()) bad("STOP takes no arguments");
                    std::lock_guard<std::mutex> lock(state_mutex);
                    if (active) cancelled.store(true, std::memory_order_relaxed);
                } else if (command == "QUIT") {
                    if (!word(rest).empty()) bad("QUIT takes no arguments");
                    break;
                } else if (command == "GEN") {
                    {
                        std::lock_guard<std::mutex> lock(state_mutex);
                        if (active) bad("engine is busy; wait for DONE before GEN");
                    }
                    auto request = parse_request(rest, context, model.geometry().vocab);
                    std::lock_guard<std::mutex> lock(state_mutex);
                    if (active) bad("engine is busy; wait for DONE before GEN");
                    pending = std::move(request);
                    active = true;
                    cancelled.store(false, std::memory_order_relaxed);
                    changed.notify_one();
                } else bad("unsupported command; expected GEN, STOP or QUIT");
            } catch (const std::exception& e) { error(e.what()); }
        }
        // The calling thread owns stdin. No detached reader remains blocked on
        // a Windows pipe when QUIT/EOF interrupts an active request.
        stop();
    }
};
} // namespace
void serve(const Options& options, Execution& model) {
    if (!options.context || options.model.empty() || options.gpu > uint32_t(std::numeric_limits<int>::max()))
        bad("invalid resident execution arguments/context");
    Resident(options, model).run();
}
} // namespace strata::muse
