package ai.wolf.autopilot;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/** GitHub API for ONE approved workflow; no arbitrary endpoints or repository writes. */
public final class GithubApi {
    private static final String BASE = "https://api.github.com/repos/19wolfalone92/Two-Worlds-Studio";
    private static final String WORKFLOW = "/actions/workflows/wolf-ai-audit.yml";
    private static final String BROWSER_BASE =
            "https://github.com/19wolfalone92/Two-Worlds-Studio/actions/runs/";

    public static final class RunInfo {
        public final long id, createdEpochMillis;
        public final String name, status, conclusion, url;
        public final boolean fromDispatch;

        public RunInfo(long id, String name, String status, String conclusion,
                       boolean fromDispatch, long createdEpochMillis) {
            this.id = id;
            this.name = name;
            this.status = status;
            this.conclusion = conclusion;
            this.fromDispatch = fromDispatch;
            this.createdEpochMillis = createdEpochMillis;
            this.url = BROWSER_BASE + id;
        }

        public String state() {
            if (!"completed".equals(status)) return TaskStore.RUNNING;
            return "success".equals(conclusion) ? TaskStore.SUCCESS : TaskStore.FAILURE;
        }
    }

    private GithubApi() {}

    private static HttpURLConnection open(String path, String method, String token) throws Exception {
        if (!path.startsWith("/actions/")) throw new IllegalArgumentException("Путь GitHub запрещён.");
        HttpURLConnection con = (HttpURLConnection) new URL(BASE + path).openConnection();
        con.setRequestMethod(method);
        con.setConnectTimeout(12000);
        con.setReadTimeout(16000);
        con.setInstanceFollowRedirects(false);
        con.setRequestProperty("Accept", "application/vnd.github+json");
        con.setRequestProperty("X-GitHub-Api-Version", "2022-11-28");
        con.setRequestProperty("User-Agent", "WolfAI-Android/1.2");
        if (token != null && !token.isEmpty())
            con.setRequestProperty("Authorization", "Bearer " + token);
        return con;
    }

    private static String body(HttpURLConnection con) throws Exception {
        InputStream input = con.getResponseCode() >= 400 ? con.getErrorStream() : con.getInputStream();
        if (input == null) return "";
        try (InputStream source = input; ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = source.read(buffer)) != -1) {
                if (out.size() + count > 250000)
                    throw new IllegalStateException("Ответ GitHub превышает допустимый размер.");
                out.write(buffer, 0, count);
            }
            return out.toString(StandardCharsets.UTF_8.name());
        }
    }

    /** 204 Accepted means only that a workflow request was accepted. */
    public static String dispatch(String project, String mode, String uid, String token) throws Exception {
        if (!CommandParser.validProject(project) || !("audit".equals(mode) || "test".equals(mode)))
            throw new IllegalArgumentException("Недопустимый тип проверки.");
        if (uid == null || !uid.matches("w[a-f0-9]{32}"))
            throw new IllegalArgumentException("Недопустимый идентификатор задачи.");
        if (token == null || token.isEmpty())
            throw new IllegalStateException("Для запуска нужен GitHub-токен с правом Actions: Read and write.");

        HttpURLConnection con = open(WORKFLOW + "/dispatches", "POST", token);
        try {
            con.setDoOutput(true);
            con.setRequestProperty("Content-Type", "application/json; charset=utf-8");
            JSONObject inputs = new JSONObject();
            inputs.put("project", project);
            inputs.put("mode", mode);
            inputs.put("client_task_id", uid);
            JSONObject payload = new JSONObject();
            payload.put("ref", "main");
            payload.put("inputs", inputs);
            byte[] data = payload.toString().getBytes(StandardCharsets.UTF_8);
            con.setFixedLengthStreamingMode(data.length);
            try (OutputStream os = con.getOutputStream()) { os.write(data); }
            int code = con.getResponseCode();
            body(con);
            if (code == 200 || code == 201 || code == 204)
                return "Задание принято GitHub. Обнови запуски, чтобы увидеть настоящий результат.";
            throw new IllegalStateException("GitHub не принял задание (HTTP " + code +
                    "). Проверь токен и права Actions.");
        } finally {
            con.disconnect();
        }
    }

    /** Legacy compatibility for future manual calls, with unique tracking key. */
    public static String dispatch(String project, String mode, String token) throws Exception {
        return dispatch(project, mode, TaskStore.newUid(), token);
    }

    public static List<RunInfo> runs(String token) throws Exception {
        HttpURLConnection con = open(WORKFLOW + "/runs?per_page=50", "GET", token);
        try {
            int code = con.getResponseCode();
            String response = body(con);
            if (code != 200)
                throw new IllegalStateException("Не удалось прочитать запуски (HTTP " + code + ").");
            JSONArray array = new JSONObject(response).getJSONArray("workflow_runs");
            List<RunInfo> runs = new ArrayList<>();
            for (int i = 0; i < array.length() && i < 50; i++) {
                JSONObject item = array.getJSONObject(i);
                long timestamp = 0;
                try {
                    timestamp = Instant.parse(item.getString("created_at")).toEpochMilli();
                } catch (Exception ignored) {
                    // Better show a run without timestamp than fake its creation date.
                }
                runs.add(new RunInfo(item.getLong("id"),
                        item.optString("display_title", "Проверка"),
                        item.optString("status", "unknown"),
                        item.isNull("conclusion") ? "pending" : item.optString("conclusion", "unknown"),
                        "workflow_dispatch".equals(item.optString("event", "")),
                        timestamp));
            }
            return runs;
        } finally {
            con.disconnect();
        }
    }

    /** Human-readable evidence from GitHub's real job/step result API. */
    public static String jobSummary(long runId, String token) throws Exception {
        if (runId <= 0) throw new IllegalArgumentException("Недопустимый запуск.");
        HttpURLConnection con = open("/actions/runs/" + runId + "/jobs?per_page=30", "GET", token);
        try {
            int code = con.getResponseCode();
            String response = body(con);
            if (code != 200) throw new IllegalStateException("Детали запуска недоступны: HTTP " + code);
            JSONArray jobs = new JSONObject(response).getJSONArray("jobs");
            StringBuilder result = new StringBuilder();
            for (int i = 0; i < jobs.length() && i < 20; i++) {
                JSONObject job = jobs.getJSONObject(i);
                result.append(job.optString("name", "job")).append(" — ")
                        .append(job.optString("conclusion", job.optString("status", "unknown")))
                        .append("\n");
                JSONArray steps = job.optJSONArray("steps");
                if (steps != null) {
                    for (int k = 0; k < steps.length() && k < 80; k++) {
                        JSONObject step = steps.getJSONObject(k);
                        String state = step.optString("conclusion", "pending");
                        if (!"success".equals(state) && !"skipped".equals(state))
                            result.append("  • ").append(step.optString("name", "step"))
                                    .append(": ").append(state).append("\n");
                    }
                }
            }
            return result.length() == 0 ? "GitHub ещё не вернул результаты шагов." : result.toString();
        } finally {
            con.disconnect();
        }
    }
}
