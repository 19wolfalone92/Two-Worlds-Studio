package ai.wolf.autopilot;

import java.util.Locale;

/** Deterministic offline parser: only selects a SAFE diagnostic recipe. */
public final class CommandParser {
    private static final String[] PROJECTS = {
            "studio", "shaira", "williams", "williams-ai", "english-child"
    };
    private CommandParser() {}

    public static boolean validProject(String project) {
        for (String item : PROJECTS) if (item.equals(project)) return true;
        return false;
    }

    public static final class Route {
        public final String project;
        public final String mode;
        Route(String project, String mode) {
            this.project = project; this.mode = mode;
        }
    }

    public static Route parse(String text, String fallbackProject, String fallbackMode) {
        String lower = (text == null ? "" : text).toLowerCase(Locale.ROOT);
        String project = fallbackProject;
        // Priority: explicit repo names before generic words.
        if (lower.contains("williams-ai") || lower.contains("williams ai") ||
                lower.contains("обучи ии") || lower.contains("ai трейдер")) {
            project = "williams-ai";
        } else if (lower.contains("шайр") || lower.contains("shaira")) {
            project = "shaira";
        } else if (lower.contains("williams") || lower.contains("торгов") ||
                lower.contains("бота") || lower.contains("бот ")) {
            project = "williams";
        } else if (lower.contains("ева") || lower.contains("eva") ||
                lower.contains("english-child") || lower.contains("english_child")) {
            project = "english-child";
        } else if (lower.contains("studio") || lower.contains("студи") ||
                lower.contains("видео") || lower.contains("story glitch") ||
                lower.contains("ролик")) {
            project = "studio";
        }
        if (!validProject(project)) project = "studio";
        String mode = fallbackMode;
        if (lower.contains("тест") || lower.contains("test") ||
                lower.contains("ошиб") || lower.contains("исправ") ||
                lower.contains("провер")) {
            mode = "test";
        }
        if (!("test".equals(mode) || "audit".equals(mode))) mode = "audit";
        return new Route(project, mode);
    }
}
