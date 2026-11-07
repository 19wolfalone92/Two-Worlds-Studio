package ai.wolf.autopilot;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * Persistent local task inbox plus real remote CI state.
 * Migration keeps all v1 tasks; a saved draft is never represented as completed.
 */
public final class TaskStore extends SQLiteOpenHelper {
    public static final String DRAFT = "draft";
    public static final String SUBMITTED = "submitted";
    public static final String RUNNING = "running";
    public static final String SUCCESS = "success";
    public static final String FAILURE = "failure";
    public static final class Task {
        public final long id;
        public final String text, project, mode, uid, status;
        public final long created, runId;
        public Task(long id, String text, String project, String mode,
                    long created, String uid, String status, long runId) {
            this.id = id; this.text = text; this.project = project; this.mode = mode;
            this.created = created; this.uid = uid; this.status = status; this.runId = runId;
        }
    }

    public TaskStore(Context context) {
        super(context, "wolf_tasks.db", null, 2);
    }

    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE tasks (" +
                "_id INTEGER PRIMARY KEY AUTOINCREMENT," +
                "body TEXT NOT NULL, project TEXT NOT NULL, " +
                "mode TEXT NOT NULL, created INTEGER NOT NULL, " +
                "task_uid TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'draft'," +
                "run_id INTEGER NOT NULL DEFAULT 0)");
    }

    @Override public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        if (oldVersion < 2) {
            db.execSQL("ALTER TABLE tasks ADD COLUMN task_uid TEXT");
            db.execSQL("ALTER TABLE tasks ADD COLUMN status TEXT NOT NULL DEFAULT 'draft'");
            db.execSQL("ALTER TABLE tasks ADD COLUMN run_id INTEGER NOT NULL DEFAULT 0");
            // Existing v1 tasks are local drafts; their old text was never dispatched.
            try (Cursor c = db.rawQuery("SELECT _id FROM tasks", null)) {
                while (c.moveToNext()) {
                    ContentValues values = new ContentValues();
                    values.put("task_uid", newUid());
                    db.update("tasks", values, "_id=?", new String[]{String.valueOf(c.getLong(0))});
                }
            }
            db.execSQL("CREATE UNIQUE INDEX tasks_uid_idx ON tasks(task_uid)");
        }
    }

    public static String newUid() {
        return "w" + UUID.randomUUID().toString().replace("-", "");
    }

    public long add(String body, String project, String mode) {
        if (body == null || body.trim().length() < 3 || body.length() > 600)
            throw new IllegalArgumentException("Задание должно содержать 3–600 символов.");
        if (!CommandParser.validProject(project) || !(mode.equals("audit") || mode.equals("test")))
            throw new IllegalArgumentException("Недопустимый проект или режим.");
        ContentValues values = new ContentValues();
        values.put("body", body.trim());
        values.put("project", project);
        values.put("mode", mode);
        values.put("created", System.currentTimeMillis());
        values.put("task_uid", newUid());
        values.put("status", DRAFT);
        long id = getWritableDatabase().insertOrThrow("tasks", null, values);
        getWritableDatabase().execSQL("DELETE FROM tasks WHERE _id NOT IN " +
                "(SELECT _id FROM tasks ORDER BY _id DESC LIMIT 100)");
        return id;
    }

    public List<Task> recent() {
        ArrayList<Task> result = new ArrayList<>();
        try (Cursor cursor = getReadableDatabase().rawQuery(
                "SELECT _id,body,project,mode,created,task_uid,status,run_id " +
                        "FROM tasks ORDER BY _id DESC LIMIT 30", null)) {
            while (cursor.moveToNext()) {
                result.add(new Task(cursor.getLong(0), cursor.getString(1),
                        cursor.getString(2), cursor.getString(3), cursor.getLong(4),
                        cursor.getString(5), cursor.getString(6), cursor.getLong(7)));
            }
        }
        return result;
    }

    public void submitted(long id, String uid) {
        ContentValues values = new ContentValues();
        values.put("status", SUBMITTED);
        getWritableDatabase().update("tasks", values, "_id=? AND task_uid=?",
                new String[]{String.valueOf(id), uid});
    }

    public int reconcile(List<GithubApi.RunInfo> runs) {
        int updated = 0;
        SQLiteDatabase db = getWritableDatabase();
        for (Task task : recent()) {
            if (DRAFT.equals(task.status)) continue;
            for (GithubApi.RunInfo run : runs) {
                if (!run.fromDispatch || !run.name.contains(task.uid)) continue;
                if (!run.name.contains(task.project) || !run.name.contains(task.mode)) continue;
                // Never associate a job that predates task creation by >120 seconds.
                if (run.createdEpochMillis > 0 && run.createdEpochMillis + 120000L < task.created) continue;
                String state = run.state();
                if (task.runId == run.id && state.equals(task.status)) break;
                ContentValues values = new ContentValues();
                values.put("status", state);
                values.put("run_id", run.id);
                updated += db.update("tasks", values, "_id=? AND task_uid=?",
                        new String[]{String.valueOf(task.id), task.uid});
                break;
            }
        }
        return updated;
    }

    public void remove(long id) {
        getWritableDatabase().delete("tasks", "_id=?", new String[]{String.valueOf(id)});
    }
}
