package ai.wolf.autopilot;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Spinner;
import android.widget.TextView;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * No paid API. Native Android front-end for free-first WOLF checks and local
 * notes; llama.cpp chat is optional and localhost-only.
 */
public final class MainActivity extends Activity {
    private static final int BG = Color.rgb(8, 14, 27);
    private static final int PANEL = Color.rgb(19, 30, 50);
    private static final int INPUT = Color.rgb(29, 42, 64);
    private static final int WHITE = Color.rgb(237, 245, 255);
    private static final int MUTED = Color.rgb(165, 181, 204);
    private static final int CYAN = Color.rgb(95, 211, 255);
    private static final int GREEN = Color.rgb(122, 231, 170);
    private static final int ORANGE = Color.rgb(255, 196, 114);
    private static final String[] PROJECT_CODES = {
            "studio", "shaira", "williams", "williams-ai", "english-child"
    };
    private static final String[] PROJECT_TITLES = {
            "🎬 STORY GLITCH Studio", "❤️ Shaira — русский язык",
            "📈 Williams Bot", "🧠 Williams AI", "📚 Eva English Child"
    };
    private static final String[] MODES = {"audit", "test"};
    private static final String[] MODE_TITLES = {"Аудит", "Тестирование"};

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private SecureStore secrets;
    private TaskStore tasks;
    private Spinner projectSpinner;
    private Spinner modeSpinner;
    private EditText commandInput;
    private EditText tokenInput;
    private EditText questionInput;
    private TextView connectionState;
    private TextView resultBox;
    private TextView aiOutput;
    private TextView aiStatus;
    private LinearLayout tasksContainer;
    private LinearLayout runsContainer;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        secrets = new SecureStore(this);
        tasks = new TaskStore(this);
        buildUi();
        updateTokenStatus();
        renderTasks();
        // Public read is possible without a token. No auto-execution on app start.
    }

    private int dp(float value) {
        return Math.round(getResources().getDisplayMetrics().density * value);
    }

    private GradientDrawable shape(int fill, float radius) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fill);
        d.setCornerRadius(dp(radius));
        return d;
    }

    private LinearLayout vertical() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    private void pad(View v, int x, int y) {
        v.setPadding(dp(x), dp(y), dp(x), dp(y));
    }

    private LinearLayout card(LinearLayout parent, String title, String subtitle) {
        LinearLayout section = vertical();
        section.setBackground(shape(PANEL, 17));
        pad(section, 17, 16);
        LinearLayout.LayoutParams outer = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        outer.setMargins(0, 0, 0, dp(15));
        parent.addView(section, outer);
        section.addView(text(title, 19, WHITE, true));
        if (subtitle != null && !subtitle.isEmpty()) {
            TextView desc = text(subtitle, 13, MUTED, false);
            add(section, desc, 0, 7);
        }
        return section;
    }

    private TextView text(String value, int size, int color, boolean bold) {
        TextView v = new TextView(this);
        v.setText(value);
        v.setTextSize(size);
        v.setTextColor(color);
        v.setLineSpacing(dp(2), 1);
        if (bold) v.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return v;
    }

    private void add(LinearLayout parent, View child, int top, int bottom) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.setMargins(0, dp(top), 0, dp(bottom));
        parent.addView(child, params);
    }

    private Button button(String label, boolean primary) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(14);
        b.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        b.setTextColor(primary ? BG : CYAN);
        b.setBackground(shape(primary ? CYAN : INPUT, 12));
        b.setMinHeight(dp(48));
        pad(b, 12, 10);
        return b;
    }

    private EditText entry(String hint, boolean multiline) {
        EditText e = new EditText(this);
        e.setTextColor(WHITE);
        e.setHintTextColor(MUTED);
        e.setHint(hint);
        e.setTextSize(15);
        e.setBackground(shape(INPUT, 12));
        e.setInputType(InputType.TYPE_CLASS_TEXT |
                (multiline ? InputType.TYPE_TEXT_FLAG_MULTI_LINE : 0));
        e.setSingleLine(!multiline);
        if (multiline) {
            e.setMinLines(2);
            e.setMaxLines(5);
            e.setGravity(Gravity.TOP | Gravity.START);
        }
        pad(e, 13, 13);
        return e;
    }

    private Spinner spinner(String[] labels) {
        Spinner s = new Spinner(this);
        ArrayAdapter<String> a = new ArrayAdapter<String>(
                this, android.R.layout.simple_spinner_item, labels) {
            @Override public View getView(int position, View convertView, ViewGroup parent) {
                TextView t = (TextView) super.getView(position, convertView, parent);
                t.setTextColor(WHITE);
                t.setTextSize(16);
                return t;
            }
            @Override public View getDropDownView(int position, View convertView, ViewGroup parent) {
                TextView t = (TextView) super.getDropDownView(position, convertView, parent);
                t.setTextColor(WHITE);
                t.setBackgroundColor(INPUT);
                pad(t, 15, 13);
                return t;
            }
        };
        a.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        s.setAdapter(a);
        s.setBackground(shape(INPUT, 11));
        s.setMinimumHeight(dp(46));
        return s;
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BG);
        LinearLayout page = vertical();
        pad(page, 16, 22);
        scroll.addView(page);
        setContentView(scroll);

        TextView brand = text("🐺 WOLF AI", 32, CYAN, true);
        add(page, brand, 0, 2);
        add(page, text("Автономный центр управления • Android v1.2", 14, MUTED, false), 0, 18);

        LinearLayout intro = card(page, "⚡ Бесплатный базовый режим",
                "Задачи хранятся на телефоне. GitHub выполняет разрешённые проверки. " +
                "Локальный ИИ необязателен. Work и платные API не нужны.");
        TextView banner = text("● OFFLINE-FIRST  •  НЕТ РЕКЛАМЫ  •  НЕТ АВТОПУБЛИКАЦИИ", 12, GREEN, true);
        add(intro, banner, 11, 0);

        LinearLayout commands = card(page, "📝 Новое задание",
                "Напиши обычным текстом. Сейчас команды классифицируются локально, " +
                "но не выполняют произвольные действия и не исправляют код автоматически.");
        commandInput = entry("Например: Проверь ошибки приложения Shaira", true);
        add(commands, commandInput, 12, 10);
        Button create = button("Добавить в локальные задачи", true);
        add(commands, create, 0, 4);
        create.setOnClickListener(v -> {
            try {
                String body = commandInput.getText().toString().trim();
                CommandParser.Route route = CommandParser.parse(body, selectedProject(), selectedMode());
                tasks.add(body, route.project, route.mode);
                commandInput.setText("");
                showResult("Задача сохранена на телефоне. Для запуска проверки нажми «Проверить» возле задачи.");
                renderTasks();
            } catch (Exception e) {
                showResult(e.getMessage());
            }
        });
        Button createAndRun = button("💥 Сохранить и запустить проверку", false);
        add(commands, createAndRun, 4, 8);
        createAndRun.setOnClickListener(v -> {
            try {
                String body = commandInput.getText().toString().trim();
                CommandParser.Route route = CommandParser.parse(body, selectedProject(), selectedMode());
                long id = tasks.add(body, route.project, route.mode);
                commandInput.setText("");
                renderTasks();
                for (TaskStore.Task task : tasks.recent()) {
                    if (task.id == id) {
                        dispatchTask(task);
                        break;
                    }
                }
            } catch (Exception e) {
                showResult("Не удалось сохранить задачу: " + e.getMessage());
            }
        });
        tasksContainer = vertical();
        add(commands, tasksContainer, 12, 0);

        LinearLayout dispatch = card(page, "⚙️ Запуск GitHub Actions",
                "Запускается существующий WOLF AI audit workflow. " +
                "Ни один проект не изменяется, сделки и публикации запрещены.");
        add(dispatch, text("Проект", 13, MUTED, true), 11, 5);
        projectSpinner = spinner(PROJECT_TITLES);
        add(dispatch, projectSpinner, 0, 7);
        add(dispatch, text("Режим", 13, MUTED, true), 3, 5);
        modeSpinner = spinner(MODE_TITLES);
        add(dispatch, modeSpinner, 0, 11);
        Button run = button("▶ Запустить проверку", true);
        add(dispatch, run, 0, 7);
        run.setOnClickListener(v -> dispatchSelected(selectedProject(), selectedMode()));
        Button refresh = button("↻ Обновить запуски", false);
        add(dispatch, refresh, 0, 9);
        refresh.setOnClickListener(v -> updateRuns());
        resultBox = text("Здесь появится результат запуска.", 13, ORANGE, false);
        add(dispatch, resultBox, 0, 12);
        runsContainer = vertical();
        add(dispatch, runsContainer, 0, 0);

        LinearLayout credentials = card(page, "🔐 Доступ к GitHub",
                "Fine-grained token только для Two-Worlds-Studio, разрешение Actions: Read and write. " +
                "Сохраняется зашифрованно через Android Keystore, не загружается в репозиторий.");
        connectionState = text("Статус токена", 13, MUTED, true);
        add(credentials, connectionState, 12, 6);
        tokenInput = entry("Вставь токен github_pat_…", false);
        tokenInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        add(credentials, tokenInput, 3, 8);
        Button save = button("Сохранить токен", true);
        add(credentials, save, 0, 7);
        save.setOnClickListener(v -> {
            try {
                secrets.save(tokenInput.getText().toString());
                tokenInput.setText("");
                updateTokenStatus();
                showResult("Токен сохранён в зашифрованном хранилище Android.");
            } catch (Exception e) {
                showResult("Не удалось сохранить токен: " + e.getMessage());
            }
        });
        Button clear = button("Удалить токен", false);
        add(credentials, clear, 0, 0);
        clear.setOnClickListener(v -> {
            secrets.clear();
            updateTokenStatus();
            showResult("Локальный GitHub-токен удалён.");
        });

        LinearLayout llm = card(page, "🧠 Локальный ИИ (опционально)",
                "Если на этом телефоне запущен llama.cpp-сервер на 127.0.0.1:8080, " +
                "можно общаться без облачного API. Модель в APK НЕ включена.");
        aiStatus = text("○ Статус локального сервера неизвестен", 13, ORANGE, true);
        add(llm, aiStatus, 11, 7);
        Button checkAi = button("↻ Проверить локальную модель", false);
        add(llm, checkAi, 0, 8);
        checkAi.setOnClickListener(v -> {
            aiStatus.setText("Подключаюсь к 127.0.0.1:8080…");
            async(LocalAi::check, value -> {
                aiStatus.setTextColor(GREEN);
                aiStatus.setText("● " + value);
            }, error -> {
                aiStatus.setTextColor(ORANGE);
                aiStatus.setText("○ Не запущена: " + error);
            });
        });
        Button guide = button("⬇️ Настройка бесплатной модели на телефоне", false);
        add(llm, guide, 0, 7);
        guide.setOnClickListener(v -> startActivity(new Intent(Intent.ACTION_VIEW,
                Uri.parse("https://github.com/19wolfalone92/Two-Worlds-Studio/tree/main/mobile/wolf-ai/termux"))));
        questionInput = entry("Вопрос для локальной модели…", true);
        add(llm, questionInput, 13, 8);
        Button ask = button("Спросить локальный ИИ", false);
        add(llm, ask, 0, 9);
        aiOutput = text("Без запущенного локального сервера этот раздел не отвечает.", 13, MUTED, false);
        add(llm, aiOutput, 0, 0);
        ask.setOnClickListener(v -> {
            String question = questionInput.getText().toString().trim();
            aiOutput.setText("Подключение к локальному ИИ…");
            async(() -> LocalAi.ask(question), answer -> aiOutput.setText(answer),
                    error -> aiOutput.setText("Локальный ИИ недоступен: " + error +
                            "\nНужен отдельно запущенный llama.cpp на 127.0.0.1:8080."));
        });

        TextView footer = text("WOLF AI 1.2 • Под контролем пользователя • Без Work", 12, MUTED, false);
        footer.setGravity(Gravity.CENTER);
        add(page, footer, 10, 22);
    }

    private String selectedProject() {
        return PROJECT_CODES[projectSpinner.getSelectedItemPosition()];
    }

    private String selectedMode() {
        return MODES[modeSpinner.getSelectedItemPosition()];
    }

    private void updateTokenStatus() {
        connectionState.setText(secrets.configured()
                ? "● Токен сохранён на устройстве"
                : "○ Токен не добавлен. Чтение публичных запусков доступно без токена.");
        connectionState.setTextColor(secrets.configured() ? GREEN : ORANGE);
    }

    private interface RemoteAction { String call() throws Exception; }
    private interface OnSuccess { void handle(String value); }
    private interface OnError { void handle(String value); }

    private void async(RemoteAction work, OnSuccess success, OnError failure) {
        executor.execute(() -> {
            try {
                String value = work.call();
                runOnUiThread(() -> { if (!isFinishing() && !isDestroyed()) success.handle(value); });
            } catch (Exception exception) {
                String reason = exception.getMessage() == null ? exception.getClass().getSimpleName() : exception.getMessage();
                runOnUiThread(() -> { if (!isFinishing() && !isDestroyed()) failure.handle(reason); });
            }
        });
    }

    private void showResult(String text) {
        if (resultBox != null) resultBox.setText(text == null ? "Ошибка без описания" : text);
    }

    private void dispatchSelected(String project, String mode) {
        showResult("Отправляю запрос на проверку " + project + "…");
        async(() -> GithubApi.dispatch(project, mode, secrets.load()),
                answer -> { showResult(answer); updateRuns(); },
                error -> showResult("Проверка не запущена: " + error));
    }

    private void dispatchTask(TaskStore.Task task) {
        if (!TaskStore.DRAFT.equals(task.status)) {
            showResult("Задание уже отправлено. Обнови статусы вместо повторного запуска.");
            return;
        }
        showResult("Отправляю задачу " + task.uid + "…");
        async(() -> GithubApi.dispatch(task.project, task.mode, task.uid, secrets.load()),
                answer -> {
                    tasks.submitted(task.id, task.uid);
                    renderTasks();
                    showResult(answer);
                    updateRuns();
                },
                error -> showResult("Задание осталось локальным: " + error));
    }

    private void showJobDetails(long runId) {
        showResult("Получаю реальные результаты шагов #" + runId + "…");
        async(() -> GithubApi.jobSummary(runId, secrets.load()),
                this::showResult,
                error -> showResult("Детали GitHub пока недоступны: " + error));
    }

    private void updateRuns() {
        showResult("Сверяю реальные запуски GitHub…");
        async(() -> {
            List<GithubApi.RunInfo> list = GithubApi.runs(secrets.load());
            int reconciled = tasks.reconcile(list);
            org.json.JSONObject info = new org.json.JSONObject();
            info.put("updated", reconciled);
            org.json.JSONArray rows = new org.json.JSONArray();
            for (GithubApi.RunInfo run : list) {
                rows.put(new org.json.JSONObject()
                        .put("id", run.id).put("name", run.name)
                        .put("status", run.status).put("conclusion", run.conclusion)
                        .put("url", run.url));
            }
            info.put("runs", rows);
            return info.toString();
        }, json -> {
            runsContainer.removeAllViews();
            renderTasks();
            try {
                org.json.JSONObject data = new org.json.JSONObject(json);
                org.json.JSONArray rows = data.getJSONArray("runs");
                showResult("Запусков: " + rows.length() +
                        " • Обновлено задач: " + data.getInt("updated") +
                        ". Нажми на запуск для деталей; удерживай для GitHub.");
                if (rows.length() == 0) add(runsContainer,
                        text("Запусков пока нет.", 13, MUTED, false), 0, 0);
                for (int i = 0; i < rows.length() && i < 15; i++) {
                    org.json.JSONObject row = rows.getJSONObject(i);
                    String status = row.getString("status");
                    String conclusion = row.getString("conclusion");
                    long runId = row.getLong("id");
                    TextView item = text("● #" + runId + " • " + status +
                            " / " + conclusion + "\n" + row.optString("name"), 13,
                            "success".equals(conclusion) ? GREEN : ORANGE, false);
                    item.setBackground(shape(INPUT, 10));
                    pad(item, 12, 11);
                    add(runsContainer, item, 0, 9);
                    String url = row.getString("url");
                    item.setOnClickListener(v -> showJobDetails(runId));
                    item.setOnLongClickListener(v -> {
                        startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
                        return true;
                    });
                }
            } catch (Exception e) {
                showResult("Ошибка разбора ответа GitHub: " + e.getMessage());
            }
        }, error -> showResult("Сверка не выполнена: " + error));
    }

    private void renderTasks() {
        if (tasksContainer == null) return;
        tasksContainer.removeAllViews();
        List<TaskStore.Task> recent = tasks.recent();
        if (recent.isEmpty()) {
            add(tasksContainer, text("Локальных задач пока нет.", 13, MUTED, false), 0, 0);
            return;
        }
        for (TaskStore.Task task : recent) {
            LinearLayout box = vertical();
            box.setBackground(shape(INPUT, 11));
            pad(box, 12, 11);
            add(tasksContainer, box, 0, 9);
            add(box, text(task.text, 14, WHITE, true), 0, 4);
            String status;
            int stateColor;
            switch (task.status) {
                case TaskStore.SUBMITTED: status = "ОТПРАВЛЕНО — ждём GitHub"; stateColor = ORANGE; break;
                case TaskStore.RUNNING: status = "ВЫПОЛНЯЕТСЯ"; stateColor = CYAN; break;
                case TaskStore.SUCCESS: status = "ПРОВЕРКИ УСПЕШНЫ"; stateColor = GREEN; break;
                case TaskStore.FAILURE: status = "ПРОВЕРКА С ОШИБКОЙ"; stateColor = ORANGE; break;
                default: status = "ЧЕРНОВИК НА ТЕЛЕФОНЕ"; stateColor = MUTED;
            }
            add(box, text(task.project + " • " + task.mode + " • " + status +
                    (task.runId > 0 ? " • #" + task.runId : ""), 12, stateColor, true), 0, 8);
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            Button check = button(TaskStore.DRAFT.equals(task.status) ? "▶ Проверить" : "↻ Статус", true);
            Button remove = button("Удалить", false);
            LinearLayout.LayoutParams half = new LinearLayout.LayoutParams(0, dp(48), 1);
            half.setMargins(0, 0, dp(7), 0);
            row.addView(check, half);
            row.addView(remove, new LinearLayout.LayoutParams(0, dp(48), 1));
            add(box, row, 0, 0);
            check.setOnClickListener(v -> {
                if (TaskStore.DRAFT.equals(task.status)) dispatchTask(task);
                else updateRuns();
            });
            remove.setOnClickListener(v -> {
                tasks.remove(task.id);
                renderTasks();
            });
        }
    }

    @Override protected void onDestroy() {
        executor.shutdownNow();
        tasks.close();
        super.onDestroy();
    }
}
