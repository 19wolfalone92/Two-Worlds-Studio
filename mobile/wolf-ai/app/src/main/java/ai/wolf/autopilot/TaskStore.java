package ai.wolf.autopilot;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.List;

/** Offline command inbox. Does NOT imply remote execution. */
public final class TaskStore extends SQLiteOpenHelper {
    public static final class Task {
        public final long id;
        public final String text;
        public final String project;
        public final String mode;
        public final long created;
        public Task(long id, String text, String project, String mode, long created) {
            this.id = id; this.text = text; this.project = project;
            this.mode = mode; this.created = created;
        }
    }

    public TaskStore(Context context) {
        super(context, "wolf_tasks.db", null, 1);
    }

    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE tasks (" +
                "_id INTEGER PRIMARY KEY AUTOINCREMENT," +
                "body TEXT NOT NULL, project TEXT NOT NULL, " +
                "mode TEXT NOT NULL, created INTEGER NOT NULL)");
    }

    @Override public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // No schema changes in v1.
    }

    public long add(String body, String project, String mode) {
        if (body == null || body.trim().length() < 3 || body.length() > 600) {
            throw new IllegalArgumentException("Задание должно содержать 3–600 символов.");
        }
        if (!CommandParser.validProject(project) || !("audit".equals(mode) || "test".equals(mode))) {
            throw new IllegalArgumentException("Недопустимый проект или режим.");
        }
        SQLiteDatabase db = getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put("body", body.trim());
        values.put("project", project);
        values.put("mode", mode);
        values.put("created", System.currentTimeMillis());
        long id = db.insertOrThrow("tasks", null, values);
        // Keep bounded storage, retain latest 100.
        db.execSQL("DELETE FROM tasks WHERE _id NOT IN " +
                "(SELECT _id FROM tasks ORDER BY _id DESC LIMIT 100)");
        return id;
    }

    public List<Task> recent() {
        ArrayList<Task> result = new ArrayList<>();
        try (Cursor cursor = getReadableDatabase().rawQuery(
                "SELECT _id,body,project,mode,created FROM tasks ORDER BY _id DESC LIMIT 30", null)) {
            while (cursor.moveToNext()) {
                result.add(new Task(cursor.getLong(0), cursor.getString(1),
                        cursor.getString(2), cursor.getString(3), cursor.getLong(4)));
            }
        }
        return result;
    }

    public void remove(long id) {
        getWritableDatabase().delete("tasks", "_id=?", new String[]{String.valueOf(id)});
    }
}
