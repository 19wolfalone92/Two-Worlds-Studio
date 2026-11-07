package ai.wolf.autopilot;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Offline-only chat transcript for the locally hosted model. No cloud sync. */
public final class ChatStore extends SQLiteOpenHelper {
    public static final class Message {
        public final String role, text;
        Message(String role, String text) {
            this.role = role; this.text = text;
        }
    }
    public ChatStore(Context context) { super(context, "wolf_chat.db", null, 1); }
    @Override public void onCreate(SQLiteDatabase db) {
        db.execSQL("CREATE TABLE messages (_id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                "role TEXT NOT NULL, body TEXT NOT NULL, created INTEGER NOT NULL)");
    }
    @Override public void onUpgrade(SQLiteDatabase db, int from, int to) {}

    public void add(String role, String body) {
        if (!("user".equals(role) || "assistant".equals(role))) throw new IllegalArgumentException("Invalid chat role");
        if (body == null || body.isEmpty()) return;
        ContentValues row = new ContentValues();
        row.put("role", role);
        row.put("body", body.substring(0, Math.min(body.length(), 6000)));
        row.put("created", System.currentTimeMillis());
        SQLiteDatabase db = getWritableDatabase();
        db.insertOrThrow("messages", null, row);
        db.execSQL("DELETE FROM messages WHERE _id NOT IN (SELECT _id FROM messages ORDER BY _id DESC LIMIT 40)");
    }

    public List<Message> recent(int max) {
        if (max < 1 || max > 20) throw new IllegalArgumentException("Invalid history limit");
        List<Message> messages = new ArrayList<>();
        try (Cursor c = getReadableDatabase().rawQuery(
                "SELECT role,body FROM messages ORDER BY _id DESC LIMIT " + max, null)) {
            while (c.moveToNext()) messages.add(new Message(c.getString(0), c.getString(1)));
        }
        Collections.reverse(messages);
        return messages;
    }

    public void clearHistory() { getWritableDatabase().delete("messages", null, null); }
}
