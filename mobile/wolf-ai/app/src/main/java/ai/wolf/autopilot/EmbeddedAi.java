package ai.wolf.autopilot;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.StatFs;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;

/**
 * Entirely embedded offline inference: native llama.cpp + bundled Qwen GGUF.
 * No Termux, no external app, no model API or external network.
 */
public final class EmbeddedAi {
    private static final String ASSET = "qwen2.5-coder-0.5b-instruct-q4_0.gguf";
    private static final String SHA256 =
            "9739055e046d62a937e5b7879012209ef40ebea8a1569a96028de491f3f091d5";
    private static final long EXPECTED_BYTES = 428_000_000L;
    private static final Object LOCK = new Object();
    private static boolean loaded = false;

    static {
        System.loadLibrary("wolf_native");
    }

    private EmbeddedAi() {}

    private static native String nativeLoad(String path);
    private static native String nativeGenerate(String prompt, int maxTokens);
    private static native void nativeUnload();

    private static String hex(byte[] data) {
        StringBuilder value = new StringBuilder();
        for (byte b : data) value.append(String.format("%02x", b & 0xff));
        return value.toString();
    }

    private static File model(Context context) throws Exception {
        File target = new File(context.getFilesDir(), ASSET);
        SharedPreferences prefs = context.getSharedPreferences("wolf_offline_ai", Context.MODE_PRIVATE);
        if (target.isFile() && target.length() >= EXPECTED_BYTES &&
                SHA256.equals(prefs.getString("verified_sha256", ""))) return target;
        prefs.edit().remove("verified_sha256").apply();

        StatFs stats = new StatFs(context.getFilesDir().getAbsolutePath());
        if (stats.getAvailableBytes() < 520_000_000L)
            throw new IllegalStateException("Для первой распаковки модели нужно ещё около 520 МБ памяти.");

        File tmp = new File(context.getFilesDir(), ASSET + ".partial");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long size = 0;
        try (InputStream in = context.getAssets().open(ASSET);
             FileOutputStream out = new FileOutputStream(tmp)) {
            byte[] bytes = new byte[1024 * 1024];
            int n;
            while ((n = in.read(bytes)) != -1) {
                digest.update(bytes, 0, n);
                out.write(bytes, 0, n);
                size += n;
                if (size > 700_000_000L) throw new IllegalStateException("Неверный размер модели.");
            }
            out.getFD().sync();
        } catch (Exception ex) {
            tmp.delete();
            throw ex;
        }
        String hash = hex(digest.digest());
        if (size < EXPECTED_BYTES || !SHA256.equals(hash)) {
            tmp.delete();
            throw new IllegalStateException("Встроенная модель не прошла SHA-256. APK повреждён.");
        }
        if (target.exists() && !target.delete())
            throw new IllegalStateException("Не удалось заменить повреждённую модель.");
        if (!tmp.renameTo(target)) {
            tmp.delete();
            throw new IllegalStateException("Не удалось сохранить модель на устройстве.");
        }
        prefs.edit().putString("verified_sha256", SHA256).apply();
        return target;
    }

    public static String check(Context context) throws Exception {
        synchronized (LOCK) {
            File source = model(context.getApplicationContext());
            if (!loaded) {
                String result = nativeLoad(source.getAbsolutePath());
                if (result.startsWith("Ошибка")) throw new IllegalStateException(result);
                loaded = true;
            }
            return "Встроенная Qwen2.5-Coder 0.5B: готова без интернета, Termux и API.";
        }
    }

    public static String ask(Context context, String question,
                             List<ChatStore.Message> history) throws Exception {
        if (question == null || question.trim().length() < 3 || question.length() > 1000)
            throw new IllegalArgumentException("Длина вопроса должна быть 3–1000 символов.");
        check(context);
        StringBuilder prompt = new StringBuilder();
        prompt.append("<|im_start|>system\n")
              .append("Ты WOLF AI, локальный помощник. Кратко отвечай по-русски. ")
              .append("У тебя нет доступа к GitHub, файлам или приложениям. ")
              .append("Не утверждай, что выполнял действия. Не проси ключи и пароли.")
              .append("<|im_end|>\n");
        if (history != null) {
            int start = Math.max(0, history.size() - 4);
            for (int i = start; i < history.size(); i++) {
                ChatStore.Message item = history.get(i);
                if (!("user".equals(item.role) || "assistant".equals(item.role))) continue;
                String text = item.text;
                if (text.length() > 250) text = text.substring(0, 250);
                text = text.replace("<|im_start|>", "").replace("<|im_end|>", "");
                prompt.append("<|im_start|>").append(item.role).append("\n")
                      .append(text).append("<|im_end|>\n");
            }
        }
        prompt.append("<|im_start|>user\n")
              .append(question.trim().replace("<|im_start|>", "").replace("<|im_end|>", ""))
              .append("<|im_end|>\n<|im_start|>assistant\n");
        return nativeGenerate(prompt.toString(), 200);
    }

    public static void unload() {
        synchronized (LOCK) {
            if (loaded) { nativeUnload(); loaded = false; }
        }
    }
}
