#include <jni.h>
#include <llama.h>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

namespace {
std::mutex guard;
llama_model * model = nullptr;
bool backend = false;
std::string loaded_path;

std::string from_jstring(JNIEnv * env, jstring text) {
    if (!text) return "";
    const char * tmp = env->GetStringUTFChars(text, nullptr);
    if (!tmp) return "";
    std::string value(tmp);
    env->ReleaseStringUTFChars(text, tmp);
    return value;
}

jstring fail(JNIEnv * env, const std::string & message) {
    return env->NewStringUTF(("Ошибка локального ИИ: " + message).c_str());
}

void fill_batch(llama_batch_ext * batch, const llama_token * tokens, int32_t size, llama_pos start) {
    llama_batch_ext_clear(batch);
    for (int32_t i = 0; i < size; i++) {
        auto index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        llama_pos position = start + i;
        llama_batch_ext_set_pos(batch, index, &position);
    }
    llama_batch_ext_set_output_logits(batch, size - 1, true);
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_wolf_autopilot_EmbeddedAi_nativeLoad(JNIEnv * env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(guard);
    std::string file = from_jstring(env, path);
    if (file.empty()) return fail(env, "Путь к модели не задан.");
    if (!backend) {
        llama_backend_init();
        backend = true;
    }
    if (model && loaded_path == file) return env->NewStringUTF("Qwen 0.5B загружена в память");
    if (model) {
        llama_model_free(model);
        model = nullptr;
        loaded_path.clear();
    }
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.use_mmap = true;
    model = llama_model_load_from_file(file.c_str(), params);
    if (!model) return fail(env, "llama.cpp не смог открыть GGUF.");
    loaded_path = file;
    return env->NewStringUTF("Qwen 0.5B загружена в память");
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_wolf_autopilot_EmbeddedAi_nativeGenerate(JNIEnv * env, jclass, jstring request, jint max_out) {
    std::lock_guard<std::mutex> lock(guard);
    if (!model) return fail(env, "Сначала открой модель.");
    std::string prompt = from_jstring(env, request);
    if (prompt.empty() || prompt.size() > 6500) return fail(env, "Слишком длинный вопрос.");
    int max_tokens = std::max(32, std::min((int) max_out, 256));
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t needed = -llama_tokenize(vocab, prompt.data(), (int32_t)prompt.size(),
                                      nullptr, 0, true, true);
    if (needed <= 0 || needed > 1300) return fail(env, "Запрос не помещается в контекст.");
    std::vector<llama_token> tokens(needed);
    if (llama_tokenize(vocab, prompt.data(), (int32_t)prompt.size(),
                       tokens.data(), needed, true, true) < 0)
        return fail(env, "Ошибка токенизации.");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) return fail(env, "Недостаточно памяти для контекста.");

    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    std::string answer;
    bool ok = true;
    int32_t pos = 0;

    for (int32_t i = 0; i < needed; i += 512) {
        int32_t chunk = std::min((int32_t)512, needed - i);
        fill_batch(batch, tokens.data() + i, chunk, pos);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { ok = false; break; }
        pos += chunk;
    }

    for (int i = 0; ok && i < max_tokens && pos < 2047; i++) {
        llama_token next = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        int need = llama_token_to_piece(vocab, next, nullptr, 0, 0, true);
        std::vector<char> buffer((size_t)std::max(64, -need + 16));
        int n = llama_token_to_piece(vocab, next, buffer.data(), (int32_t)buffer.size(), 0, true);
        if (n >= 0) answer.append(buffer.data(), n);
        fill_batch(batch, &next, 1, pos);
        ok = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
        pos++;
    }
    llama_batch_ext_free(batch);
    llama_sampler_free(sampler);
    llama_free(ctx);
    if (!ok) return fail(env, "Ошибка вычисления модели.");
    if (answer.empty()) return env->NewStringUTF("Модель не сформировала ответ. Попробуй короткий запрос.");
    return env->NewStringUTF(answer.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_ai_wolf_autopilot_EmbeddedAi_nativeUnload(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(guard);
    if (model) { llama_model_free(model); model = nullptr; loaded_path.clear(); }
}
