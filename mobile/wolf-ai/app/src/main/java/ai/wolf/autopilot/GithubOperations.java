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
 * Allowlisted GitHub Actions workflows for non-publishing Studio jobs.
 * No arbitrary URLs, workflow IDs, repository writes, or user shell commands.
 */
public final class GithubOperations {
    private static final String ROOT = "https://api.github.com/repos/19wolfalone92/Two-Worlds-Studio/actions";
    private static final String BROWSER = "https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/";
    public static final String APK = "android";
    public static final String STUDIO = "studio";
    public static final String FREE_TESTS = "provider-tests";

    private GithubOperations() {}

    private static String workflow(String kind) {
        switch (kind) {
            case APK: return "wolf-ai-android.yml";
            case STUDIO: return "ci.yml";
            case FREE_TESTS: return "studio-v3-free-tests.yml";
            default: throw new IllegalArgumentException("Запрещённый тип задания.");
        }
    }

    private static HttpURLConnection connect(String kind, String suffix, String method, String token) throws Exception {
        URL url = new URL(ROOT + "/workflows/" + workflow(kind) + suffix);
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setRequestMethod(method);
        c.setInstanceFollowRedirects(false);
        c.setConnectTimeout(12000);
        c.setReadTimeout(16000);
        c.setRequestProperty("Accept", "application/vnd.github+json");
        c.setRequestProperty("X-GitHub-Api-Version", "2022-11-28");
        c.setRequestProperty("User-Agent", "WolfAI-Android/1.3");
        if (token != null && !token.isEmpty()) c.setRequestProperty("Authorization", "Bearer " + token);
        return c;
    }

    private static String read(HttpURLConnection c) throws Exception {
        InputStream stream = c.getResponseCode() >= 400 ? c.getErrorStream() : c.getInputStream();
        if (stream == null) return "";
        try (InputStream in = stream; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] bytes = new byte[4096];
            int n;
            while ((n = in.read(bytes)) != -1) {
                if (out.size() + n > 160000) throw new IllegalStateException("GitHub ответил слишком большим файлом.");
                out.write(bytes, 0, n);
            }
            return out.toString(StandardCharsets.UTF_8.name());
        }
    }

    public static String dispatch(String kind, String token) throws Exception {
        if (token == null || token.isEmpty())
            throw new IllegalStateException("Добавь в приложении GitHub PAT с Actions: Read and write.");
        HttpURLConnection c = connect(kind, "/dispatches", "POST", token);
        try {
            c.setDoOutput(true);
            c.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            JSONObject payload = new JSONObject().put("ref", "main");
            // Studio only renders an engineering preview, never publishes.
            if (STUDIO.equals(kind)) {
                payload.put("inputs", new JSONObject().put("episode_file",
                        "episodes/E01_THE_REFLECTION.json"));
            }
            byte[] data = payload.toString().getBytes(StandardCharsets.UTF_8);
            c.setFixedLengthStreamingMode(data.length);
            try (OutputStream out = c.getOutputStream()) { out.write(data); }
            int status = c.getResponseCode();
            read(c); // Drain and drop response bodies (never echo credentials).
            if (status != 204 && status != 201 && status != 200)
                throw new IllegalStateException("GitHub отклонил запуск HTTP " + status + ".");
            return "Задание принято GitHub. Это ещё не готовая сборка. Используй «Состояние производства».";
        } finally {
            c.disconnect();
        }
    }

    /** Status of last 3 actual workflow runs, including direct download navigation. */
    public static String report(String kind, String token) throws Exception {
        HttpURLConnection c = connect(kind, "/runs?per_page=3", "GET", token);
        try {
            int status = c.getResponseCode();
            String body = read(c);
            if (status != 200) throw new IllegalStateException("GitHub HTTP " + status + ".");
            JSONArray runs = new JSONObject(body).getJSONArray("workflow_runs");
            StringBuilder sb = new StringBuilder();
            sb.append("Процесс: ").append(kind).append("\n");
            if (runs.length() == 0) return sb.append("Запусков пока нет.").toString();
            for (int i = 0; i < runs.length() && i < 3; i++) {
                JSONObject r = runs.getJSONObject(i);
                long id = r.getLong("id");
                String conclusion = r.optString("conclusion", "pending");
                if (r.isNull("conclusion")) conclusion = "ещё выполняется";
                sb.append("#").append(id).append(" — ")
                        .append(r.optString("status", "?")).append(" / ")
                        .append(conclusion).append("\n")
                        .append(BROWSER).append(id).append("\n");
            }
            sb.append("APK/MP4/отчёты находятся в Artifacts соответствующего запуска.");
            return sb.toString();
        } finally {
            c.disconnect();
        }
    }
}
