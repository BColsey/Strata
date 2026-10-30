// Shared token-ID CLI support; numerical graphs live in separate executables.
#include "strata/muse/model.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>

namespace strata::muse {
namespace {
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("Muse: " + s); }
}

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) fail("size arithmetic overflow");
    return a + b;
}
size_t checked_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a) fail("size arithmetic overflow");
    return a * b;
}
uint32_t parse_uint(const std::string& s, const char* name) {
    if (s.empty()) fail(std::string("empty ") + name);
    uint32_t n = 0;
    for (char c : s) {
        if (c < '0' || c > '9' || n > (UINT32_MAX - uint32_t(c - '0')) / 10)
            fail(std::string("invalid/out-of-range ") + name);
        n = n * 10 + uint32_t(c - '0');
    }
    return n;
}
std::vector<uint32_t> parse_tokens(const std::string& s) {
    if (s.size() > 16 * 1024 * 1024) fail("token list too large");
    std::vector<uint32_t> out;
    size_t start = 0;
    do {
        const size_t end = s.find(',', start);
        out.push_back(parse_uint(s.substr(start, end == std::string::npos ? end : end - start), "token"));
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return out;
}
Options parse_options(int argc, const char* const* argv, bool llama) {
    Options o; std::set<std::string> seen;
    if (llama) o.backend = "llama-cpu";
    for (int i = 1; i < argc; ++i) {
        std::string key(argv[i]);
        if (key == "--max-context") key = "--ctx";
        if (!seen.insert(key).second) fail("duplicate argument " + key);
        if (key == "--serve") { o.serve = true; continue; }
        if (key != "--model" && key != "--tokens" && key != "--ctx" && key != "--steps" &&
            key != "--backend" && key != "--gpu" && key != "--logits-file" &&
            !(llama && (key == "--moe-cache-mib" || key == "--threads"))) fail("unknown argument " + key);
        if (++i == argc) fail("missing value for " + key);
        const std::string v(argv[i]);
        if (key == "--model") o.model = v;
        else if (key == "--tokens") o.tokens = parse_tokens(v);
        else if (key == "--ctx") o.context = parse_uint(v, "context");
        else if (key == "--steps") o.steps = parse_uint(v, "steps");
        else if (key == "--gpu") {
            if (llama) o.gpus = parse_tokens(v);
            else o.gpu = parse_uint(v, "GPU");
        }
        else if (key == "--moe-cache-mib") o.moe_cache_mib = parse_uint(v, "expert cache MiB");
        else if (key == "--threads") {
            o.threads = parse_uint(v, "threads");
            if (!o.threads || o.threads > 1024) fail("threads must be in 1..1024");
        }
        else if (key == "--backend") o.backend = v;
        else o.logits_file = v;
    }
    if (o.serve) {
        if (o.model.empty()) fail("--model is required");
        if (seen.count("--tokens") || seen.count("--steps") || seen.count("--logits-file"))
            fail("--serve cannot be combined with --tokens, --steps or --logits-file");
    } else if (o.model.empty() || o.logits_file.empty() || o.tokens.empty())
        fail("--model, --tokens and --logits-file are required");
    if (!o.context || o.tokens.size() > o.context || o.steps > o.context - o.tokens.size())
        fail("prompt plus generation exceeds context");
    if (llama) {
        if (o.backend != "llama-cpu" && o.backend != "llama-cuda") fail("backend must be llama-cpu or llama-cuda");
        if (o.context > uint32_t(INT32_MAX)) fail("context exceeds position range");
        if (o.backend == "llama-cpu" && (seen.count("--gpu") || seen.count("--moe-cache-mib")))
            fail("--gpu and --moe-cache-mib require llama-cuda");
        if (o.backend == "llama-cuda") {
            if (o.gpus.empty()) o.gpus = {0};
            if (!seen.count("--moe-cache-mib")) o.moe_cache_mib = 6144;
        }
        if (o.gpus.size() > 64 || std::set<uint32_t>(o.gpus.begin(), o.gpus.end()).size() != o.gpus.size())
            fail("GPU list must contain at most 64 distinct devices");
        for (auto gpu : o.gpus) if (gpu > uint32_t(INT32_MAX)) fail("GPU index out of range");
        checked_mul(o.moe_cache_mib, size_t(1024 * 1024));
    } else if (o.backend != "cpu" && o.backend != "cuda") fail("backend must be cpu or cuda");
    if (o.gpu > uint32_t(std::numeric_limits<int>::max())) fail("GPU index out of range");
    return o;
}
size_t Geometry::memory_bytes(size_t context) const {
    if (!context || context > train_context || !layers || !hidden || !vocab || !heads || !kv_heads ||
        !head_dim || !latent || !experts || !top_k || ff.size() != layers || local.size() != layers)
        fail("invalid context/geometry");
    const size_t qwidth = checked_mul(heads, head_dim), kvwidth = checked_mul(kv_heads, head_dim);
    size_t width = std::max({hidden, vocab, qwidth, kvwidth, latent, experts});
    for (auto n : ff) { if (!n) fail("zero expert FF dimension"); width = std::max(width, n); }
    if (width > tile_bytes / sizeof(float)) fail("geometry exceeds bounded row/activation size");
    size_t kv = 0;
    for (size_t i = 0; i < layers; ++i)
        kv = checked_add(kv, checked_mul(checked_mul(local[i] ? std::min(context, window) : context, kvwidth), 2));
    // Conservative bound: live vector temporaries, indices, attention scores,
    // full logits, and one decoded tile. Output is streamed, not stored per token.
    const size_t scratch = checked_add(tile_bytes, checked_mul(checked_add(checked_mul(width, 64), context), sizeof(float)));
    const size_t bytes = checked_add(checked_mul(kv, sizeof(float)), scratch);
    if (bytes > host_budget_bytes) fail("KV plus scratch exceeds 1 GiB reference budget");
    return bytes;
}

void run(const Options& o, Execution& model) {
    if (o.model.empty() || o.logits_file.empty() || o.tokens.empty() || !o.context ||
        o.tokens.size() > o.context || o.steps > o.context - o.tokens.size() || o.gpu > uint32_t(INT32_MAX))
        fail("invalid execution arguments/context");
    for (auto t : o.tokens) if (t >= model.geometry().vocab) fail("token outside vocabulary");
    // Never overwrite an input shard with output, including equivalent/symlink paths.
    const auto paths = gguf_split_paths(o.model);
    for (const auto& p : paths) {
        std::error_code ec;
        if (std::filesystem::equivalent(p, o.logits_file, ec)) fail("output aliases a model shard");
    }
    std::ofstream out(o.logits_file, std::ios::trunc);
    if (!out) fail("cannot open logits file");
    out << std::setprecision(std::numeric_limits<float>::max_digits10) << "{\"tokens\":[";
    for (size_t i = 0; i < o.tokens.size(); ++i) { if (i) out << ','; out << o.tokens[i]; }
    out << "],\"logits\":[";
    std::vector<float> logits;
    for (size_t i = 0; i < o.tokens.size(); ++i) {
        logits = model.forward(o.tokens[i], {});
        if (i) out << ',';
        out << '[';
        for (size_t j = 0; j < logits.size(); ++j) { if (j) out << ','; out << logits[j]; }
        out << ']';
        if (!out) fail("cannot write logits file");
    }
    out << "],\"generated_tokens\":[";
    for (size_t i = 0; i < o.steps; ++i) {
        const auto token = uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (i) out << ',';
        out << token;
        if (model.is_end(token)) break;
        if (i + 1 < o.steps) logits = model.forward(token, {});
    }
    out << "]}\n"; out.close();
    if (!out) fail("cannot finish logits file");
}
} // namespace strata::muse
