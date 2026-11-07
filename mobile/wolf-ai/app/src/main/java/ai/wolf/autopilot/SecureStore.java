package ai.wolf.autopilot;

import android.content.Context;
import android.content.SharedPreferences;
import android.security.keystore.KeyGenParameterSpec;
import android.security.keystore.KeyProperties;
import android.util.Base64;

import java.nio.charset.StandardCharsets;
import java.security.KeyStore;

import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import javax.crypto.spec.GCMParameterSpec;

/** Encrypts the GitHub token with a non-exportable Android Keystore AES key. */
public final class SecureStore {
    private static final String ALIAS = "WOLF_GITHUB_PAT_AES_V1";
    private static final String PREFS = "wolf_secure_v1";
    private static final String CIPHERTEXT = "token";
    private static final String NONCE = "nonce";
    private static final String TRANSFORMATION = "AES/GCM/NoPadding";

    private final SharedPreferences prefs;

    public SecureStore(Context context) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    private SecretKey key() throws Exception {
        KeyStore store = KeyStore.getInstance("AndroidKeyStore");
        store.load(null);
        KeyStore.Entry existing = store.getEntry(ALIAS, null);
        if (existing instanceof KeyStore.SecretKeyEntry) {
            return ((KeyStore.SecretKeyEntry) existing).getSecretKey();
        }
        KeyGenerator generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore");
        generator.init(new KeyGenParameterSpec.Builder(ALIAS,
                KeyProperties.PURPOSE_ENCRYPT | KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build());
        return generator.generateKey();
    }

    public void save(String token) throws Exception {
        String value = token == null ? "" : token.trim();
        if (value.isEmpty() || value.length() > 512 || !value.matches("[A-Za-z0-9_]+")) {
            throw new IllegalArgumentException("Некорректный формат GitHub-токена.");
        }
        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.ENCRYPT_MODE, key());
        byte[] data = cipher.doFinal(value.getBytes(StandardCharsets.UTF_8));
        prefs.edit()
                .putString(CIPHERTEXT, Base64.encodeToString(data, Base64.NO_WRAP))
                .putString(NONCE, Base64.encodeToString(cipher.getIV(), Base64.NO_WRAP))
                .apply();
    }

    public String load() throws Exception {
        String value = prefs.getString(CIPHERTEXT, null);
        String iv = prefs.getString(NONCE, null);
        if (value == null || iv == null) return "";
        Cipher cipher = Cipher.getInstance(TRANSFORMATION);
        cipher.init(Cipher.DECRYPT_MODE, key(),
                new GCMParameterSpec(128, Base64.decode(iv, Base64.DEFAULT)));
        return new String(cipher.doFinal(Base64.decode(value, Base64.DEFAULT)), StandardCharsets.UTF_8);
    }

    public boolean configured() {
        return prefs.contains(CIPHERTEXT) && prefs.contains(NONCE);
    }

    public void clear() {
        prefs.edit().remove(CIPHERTEXT).remove(NONCE).apply();
    }
}
