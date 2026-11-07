package ai.wolf.autopilot;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/** Fixed-endpoint GitHub Actions client: no arbitrary URLs or repository writes. */
public final class GithubApi {
    private static final String BASE = "https://api.github.com/repos/19wolfalone92/Two-Worlds-Studio";
    private static final String WORKFLOW = "/actions/workflows/wolf-ai-audit.yml";
    private static final String BROWSER_BASE =
            "https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/";

    public static final class RunInfo {
        public final long id;
        public final String name;
        public final String status;
        public final String conclusion;
        public final String url;
        public RunInfo(long id, String name, String status, String conclusion) {
            this.id = id;
            this.name = name;
            this.status = status;
            this.conclusion = conclusion;
            this.url = BROWSER_BASE + id;
        }
    }

    private GithubApi() {}

    private static HttpURLConnection open(String path, String method, String token) throws Exception {
        HttpURLConnection con = (HttpURLConnection) new URL(BASE + path).openConnection();
        con.setRequestMethod(method);
        con.setConnectTimeout(12000);
        con.setReadTimeout(16000);
        con.setRequestProperty("Accept", "application/vnd.github+json");
        con.setRequestProperty("X-GitHub-Api-Version", "2022-11-28");
        con.setRequestProperty("User-Agent", "WolfAI-Android/1.0");
        if (token != null && !token.isEmpty()) {
            con.setRequestProperty("Authorization", "Bearer " + token);
        }
        return con;
    }

    private static String body(HttpURLConnection con) throws Exception {
        InputStream input = con.getResponseCode() >= 400 ? con.getErrorStream() : con.getInputStream();
        if (input == null) return "";
        try (InputStream source = input; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = source.read(buffer)) != -1) {
                if (out.size() + count > 200000) {
                    throw new IllegalStateException("Ответ GitHub превышает допустимый размер.");
                }
                out.write(buffer, 0, count);
            }
            return out.toString(StandardCharsets.UTF_8.name());
        }
    }

    public static String dispatch(String project, String mode, String token) throws Exception {
        if (!CommandParser.validProject(project) ||
                !("audit".equals(mode) || "test".equals(mode))) {
            throw new IllegalArgumentException("Недопустимый тип проверки.");
        }
        if (token == null || token.isEmpty()) {
            throw new IllegalStateException("Нужен GitHub-токен с правом Actions: read/write.");
        }
        HttpURLConnection con = open(WORKFLOW + "/dispatches", "POST", token);
        try {
            con.setDoOutput(true);
            con.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            JSONObject inputs = new JSONObject();
            inputs.put("project", project);
            inputs.put("mode", mode);
            JSONObject payload = new JSONObject();
            payload.put("ref", "main");
            payload.put("inputs", inputs);
            byte[] data = payload.toString().getBytes(StandardCharsets.UTF_8);
            con.setFixedLengthStreamingMode(data.length);
            try (OutputStream os = con.getOutputStream()) { os.write(data); }
            int status = con.getResponseCode();
            if (status == 200 || status == 201 || status == 204) {
                return "GitHub принял запрос на проверку " + project
                        + ". Это НЕ означает, что тесты уже завершились. Обнови список запусков.";
            }
            body(con);  // Drain; never display API error bodies with metadata.
            throw new IllegalStateException("GitHub не принял запуск (HTTP " + status +
                    "). Проверь права Actions: read/write и доступ к репозиторию.");
        } finally {
            con.disconnect();
        }
    }

    public static List<RunInfo> runs(String token) throws Exception {
        HttpURLConnection con = open(WORKFLOW + "/runs?per_page=10", "GET", token);
        try {
            int status = con.getResponseCode();
            String response = body(con);
            if (status != 200) throw new IllegalStateException("Не удалось прочитать запуски (HTTP " + status + ").");
            JSONArray array = new JSONObject(response).getJSONArray("workflow_runs");
            List<RunInfo> runs = new ArrayList<>();
            for (int i = 0; i < array.length() && i < 10; i++) {
                JSONObject item = array.getJSONObject(i);
                long id = item.getLong("id");
                String conclusion = item.isNull("conclusion") ? "ещё не завершён" : item.optString("conclusion", "unknown");
                runs.add(new RunInfo(id, item.optString("display_title", "Проверка"),
                        item.optString("status", "unknown"), conclusion));
            }
            return runs;
        } finally {
            con.disconnect();
        }
    }
}
