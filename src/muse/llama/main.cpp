// Opt-in, separate libllama process: never linked into Strata's Qwen engine.
#include "strata/muse/model.hpp"
#include "llama.h"
#include "gguf.h"
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

namespace {
using namespace strata::muse;
[[noreturn]] void fail(const std::string& s) { throw std::runtime_error("Muse llama: " + s); }
struct Backend {
    Backend() { ggml_backend_load_all(); llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

class LlamaModel final : public Execution {
    Backend backend;
    std::vector<ggml_backend_dev_t> devices;
    std::vector<float> split;
    llama_model_tensor_buft_override overrides[2]{};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    Geometry g;
    size_t limit, position = 0;
    std::set<uint32_t> end_tokens;
    const std::function<void()>* checkpoint = nullptr;
    std::exception_ptr interrupted;

    // Never unwind a C++ exception through a C backend callback.
    static bool abort(void* opaque) noexcept {
        auto& self = *static_cast<LlamaModel*>(opaque);
        if (self.interrupted) return true;
        try { if (self.checkpoint && *self.checkpoint) (*self.checkpoint)(); }
        catch (...) { self.interrupted = std::current_exception(); return true; }
        return false;
    }
public:
    explicit LlamaModel(const Options& o) : limit(o.context) {
        std::unique_ptr<gguf_context, decltype(&gguf_free)> header(
            gguf_init_from_file(o.model.c_str(), {true, nullptr}), gguf_free);
        if (!header) fail("cannot read GGUF metadata");
        const auto arch = gguf_find_key(header.get(), "general.architecture");
        if (arch < 0 || gguf_get_kv_type(header.get(), arch) != GGUF_TYPE_STRING ||
            std::string(gguf_get_val_str(header.get(), arch)) != "muse-spark")
            fail("architecture must be muse-spark");
        const auto context = gguf_find_key(header.get(), "muse-spark.context_length");
        if (context < 0 || gguf_get_kv_type(header.get(), context) != GGUF_TYPE_UINT32)
            fail("missing/invalid trained context");
        g.train_context = gguf_get_val_u32(header.get(), context);
        if (!limit || limit > g.train_context) fail("context exceeds trained context");
        for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"}) {
            const auto id = gguf_find_key(header.get(), key);
            if (id < 0) continue;
            if (gguf_get_kv_type(header.get(), id) != GGUF_TYPE_UINT32) fail("invalid EOS/EOT metadata");
            end_tokens.insert(gguf_get_val_u32(header.get(), id));
        }
        header.reset();

        auto mp = llama_model_default_params();
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        mp.use_extra_bufts = false;
        mp.n_gpu_layers = o.backend == "llama-cuda" ? INT32_MAX : 0;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        split.assign(llama_max_devices(), 0.0f);
        if (o.gpus.size() > split.size()) fail("too many GPU devices for this build");
        for (auto gpu : o.gpus) {
            const auto name = "CUDA" + std::to_string(gpu);
            const auto device = ggml_backend_dev_by_name(name.c_str());
            if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU)
                fail("requested device unavailable: " + name + "; check CUDA build and CUDA_VISIBLE_DEVICES");
            split[devices.size()] = 1.0f;
            devices.push_back(device);
            std::cerr << "Muse llama device: " << name << " " << ggml_backend_dev_description(device) << '\n';
        }
        devices.push_back(nullptr);
        mp.devices = devices.data();
        mp.tensor_split = split.data();
        const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) fail("CPU backend unavailable");
        // Same selected-expert cache placement as the independently tested backend.
        overrides[0] = {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_dev_buffer_type(cpu)};
        if (o.backend == "llama-cuda") mp.tensor_buft_overrides = overrides;
        model.reset(llama_model_load_from_file(o.model.c_str(), mp));
        if (!model) fail("model load failed; see backend diagnostics above");
        const auto vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        if (vocab <= 0) fail("empty vocabulary");
        g.vocab = size_t(vocab);
        for (auto id : end_tokens) if (id >= g.vocab) fail("EOS/EOT outside vocabulary");

        auto cp = llama_context_default_params();
        cp.n_ctx = o.context;
        cp.n_batch = cp.n_ubatch = cp.n_seq_max = 1;
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.moe_cache_size = checked_mul(o.moe_cache_mib, size_t(1024 * 1024));
        cp.abort_callback = abort;
        cp.abort_callback_data = this;
        if (o.threads) cp.n_threads = cp.n_threads_batch = int(o.threads);
        ctx.reset(llama_init_from_model(model.get(), cp));
        if (!ctx) fail("context/cache allocation failed; reduce context or --moe-cache-mib");
        std::cerr << "Muse llama: requested_context=" << limit << " allocated_context=" << llama_n_ctx(ctx.get())
                  << " expert_cache_mib=" << o.moe_cache_mib << " threads=" << cp.n_threads
                  << " pin=e09a78a25dec3f42175ad59ecc540a54bed0b9b2 library=" << llama_version() << '\n';
    }
    const Geometry& geometry() const override { return g; }
    bool is_end(uint32_t token) const override { return end_tokens.count(token) != 0; }
    void reset() override {
        llama_synchronize(ctx.get());
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        position = 0;
        interrupted = nullptr;
    }
    std::vector<float> forward(uint32_t token, const std::function<void()>& check) override {
        if (token >= g.vocab || position >= limit) fail("token/context out of range");
        if (check) check();
        struct Clear {
            LlamaModel& self;
            ~Clear() { self.checkpoint = nullptr; self.interrupted = nullptr; }
        } clear{*this};
        checkpoint = &check;
        interrupted = nullptr;
        auto input = llama_token(token);
        const int status = llama_decode(ctx.get(), llama_batch_get_one(&input, 1));
        llama_synchronize(ctx.get());
        if (interrupted) std::rethrow_exception(interrupted);
        // CUDA can finish the current forward before noticing STOP. Never publish
        // its token after cancellation; reset KV before the next request.
        if (check) check();
        if (status != 0) fail("decode failed (status " + std::to_string(status) + ")");
        const auto* logits = llama_get_logits_ith(ctx.get(), -1);
        if (!logits) fail("missing logits");
        std::vector<float> result(logits, logits + g.vocab);
        for (float value : result) if (!std::isfinite(value)) fail("nonfinite logits");
        ++position;
        return result;
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = strata::muse::parse_options(argc, argv, true);
        LlamaModel model(options);
        if (options.serve) strata::muse::serve(options, model);
        else strata::muse::run(options, model);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
