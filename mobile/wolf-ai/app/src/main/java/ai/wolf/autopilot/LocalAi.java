package ai.wolf.autopilot;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Optional llama.cpp / OpenAI-compatible endpoint on the SAME Android device.
 * Never calls external provider APIs or transmits GitHub credentials.
 */
public final class LocalAi {
    private static final String BASE = "http://127.0.0.1:8080/v1";
    private LocalAi() {}

    private static HttpURLConnection connection(String path, String method) throws Exception {
        HttpURLConnection con = (HttpURLConnection) new URL(BASE + path).openConnection();
        con.setRequestMethod(method);
        con.setConnectTimeout(4000);
        con.setReadTimeout(60000);
        con.setRequestProperty("Accept", "application/json");
        return con;
    }

    private static String read(HttpURLConnection con) throws Exception {
        InputStream stream = con.getResponseCode() >= 400 ? con.getErrorStream() : con.getInputStream();
        if (stream == null) return "";
        try (InputStream in = stream; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) > 0) {
                if (out.size() + n > 250000) throw new IllegalStateException("Слишком большой ответ ИИ.");
                out.write(buf, 0, n);
            }
            return out.toString(StandardCharsets.UTF_8.name());
        }
    }

    /** Real readiness check; does not download a model or claim to have one. */
    public static String check() throws Exception {
        HttpURLConnection models = connection("/models", "GET");
        try {
            int status = models.getResponseCode();
            if (status != 200) throw new IllegalStateException("Сервер не отвечает (HTTP " + status + ")");
            JSONObject info = new JSONObject(read(models));
            JSONArray data = info.getJSONArray("data");
            if (data.length() == 0) throw new IllegalStateException("Сервер запущен, но модель не загружена.");
            return "Модель доступна: " + data.getJSONObject(0).optString("id", "local");
        } finally {
            models.disconnect();
        }
    }

    public static String ask(String question) throws Exception {
        return ask(question, java.util.Collections.emptyList());
    }

    public static String ask(String question, java.util.List<ChatStore.Message> history) throws Exception {
        if (question == null || question.trim().length() < 3 || question.length() > 1500)
            throw new IllegalArgumentException("Вопрос должен содержать 3–1500 символов.");

        HttpURLConnection models = connection("/models", "GET");
        String modelName;
        try {
            if (models.getResponseCode() != 200)
                throw new IllegalStateException("Локальная модель пока не запущена.");
            JSONObject info = new JSONObject(read(models));
            JSONArray data = info.getJSONArray("data");
            if (data.length() == 0) throw new IllegalStateException("Локальная модель не загружена.");
            modelName = data.getJSONObject(0).getString("id");
        } finally {
            models.disconnect();
        }
        JSONObject payload = new JSONObject();
        payload.put("model", modelName);
        payload.put("max_tokens", 384);
        payload.put("stream", false);
        JSONArray messages = new JSONArray();
        messages.put(new JSONObject().put("role", "system").put("content",
                "Ты WOLF AI — локальный помощник. Отвечай по-русски, кратко и точно. " +
                "Не утверждай, что ты изменил GitHub или выполнил тесты. " +
                "У тебя нет инструментов изменения файлов. Никогда не запрашивай секреты."));
        // Keep a tiny bounded conversation window for 0.5B GGUF models
        // running with a 1536-token context limit. Localhost only.
        if (history != null) {
            int start = Math.max(0, history.size() - 4);
            for (int i = start; i < history.size(); i++) {
                ChatStore.Message previous = history.get(i);
                if (!("user".equals(previous.role) || "assistant".equals(previous.role))) continue;
                String bounded = previous.text.length() > 340
                        ? previous.text.substring(0, 340) : previous.text;
                messages.put(new JSONObject().put("role", previous.role)
                        .put("content", bounded));
            }
        }
        messages.put(new JSONObject().put("role", "user").put("content", question.trim()));
        payload.put("messages", messages);

        HttpURLConnection con = connection("/chat/completions", "POST");
        try {
            con.setDoOutput(true);
            con.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            byte[] bytes = payload.toString().getBytes(StandardCharsets.UTF_8);
            con.setFixedLengthStreamingMode(bytes.length);
            try (OutputStream out = con.getOutputStream()) { out.write(bytes); }
            int code = con.getResponseCode();
            String response = read(con);
            if (code != 200) throw new IllegalStateException("Локальный ИИ ответил с ошибкой HTTP " + code + ".");
            return new JSONObject(response).getJSONArray("choices").getJSONObject(0)
                    .getJSONObject("message").optString("content", "Модель вернула пустой ответ.");
        } finally {
            con.disconnect();
        }
    }
}
