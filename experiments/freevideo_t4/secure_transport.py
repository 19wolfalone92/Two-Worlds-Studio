"""Encrypted private-photo transport for isolated Freevideo/Kaggle experiments.
The only public GitHub payload is AES-256-GCM ciphertext, never plaintext media.
Recipient X25519 key is derived on GitHub Actions from an existing private Kaggle
token; that token is never printed, read by the assistant, or stored in Git.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

CONTEXT = b'Freevideo/photo-to-private-Kaggle/v1'
SALT = b'Freevideo/X25519/GitHub-Actions/Kaggle-Token/v1'
MAGIC = 'FREEVIDEO_ENCRYPTED_IMAGE_V1'

def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode('ascii')

def _unb64(value: str, expected: int | None = None) -> bytes:
    if not isinstance(value, str) or len(value) > 2000000:
        raise ValueError('Invalid base64 field length')
    raw = base64.b64decode(value, validate=True)
    if expected is not None and len(raw) != expected:
        raise ValueError('Invalid binary field size')
    return raw

def _private(token: str) -> X25519PrivateKey:
    if not isinstance(token, str) or len(token) < 12 or len(token) > 4096 or '\n' in token:
        raise ValueError('Missing/invalid private Kaggle credential')
    seed = HKDF(algorithm=hashes.SHA256(), length=32, salt=SALT,
                info=CONTEXT+b'/recipient').derive(token.encode('utf-8'))
    return X25519PrivateKey.from_private_bytes(seed)

def public_key(token: str) -> str:
    return _b64(_private(token).public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw))

def encrypt(media: bytes, recipient: str, *, output_public: str) -> dict:
    if not 2048 <= len(media) <= 8*1024*1024:
        raise ValueError('Invalid media length')
    jpeg = media.startswith(bytes.fromhex('ffd8ff'))
    png = media.startswith(bytes.fromhex('89504e470d0a1a0a'))
    if not jpeg and not png:
        raise ValueError('Only JPEG or PNG media permitted')
    recipient_key = X25519PublicKey.from_public_bytes(_unb64(recipient,32))
    output_key = _unb64(output_public,32)
    ephemeral = X25519PrivateKey.generate()
    salt, nonce = os.urandom(16), os.urandom(12)
    shared = ephemeral.exchange(recipient_key)
    aes_key = HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,
                   info=CONTEXT+b'/image').derive(shared)
    digest = hashlib.sha256(media).hexdigest()
    aad = MAGIC.encode()+b'|'+digest.encode()+b'|'+output_key
    ciphertext = AESGCM(aes_key).encrypt(nonce, media, aad)
    return dict(format=MAGIC,source_sha256=digest,
                ephemeral_public_key=_b64(ephemeral.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,format=serialization.PublicFormat.Raw)),
                salt=_b64(salt),nonce=_b64(nonce),ciphertext=_b64(ciphertext),
                output_public_key=output_public,source_format='jpeg' if jpeg else 'png')

def decrypt(payload: dict, token: str) -> bytes:
    keys={'format','source_sha256','ephemeral_public_key','salt','nonce',
          'ciphertext','output_public_key','source_format'}
    if not isinstance(payload,dict) or set(payload)!=keys or payload['format'] != MAGIC:
        raise ValueError('Invalid encrypted payload schema')
    digest=payload['source_sha256']
    if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
        raise ValueError('Invalid pinned source hash')
    output_public=_unb64(payload['output_public_key'],32)
    salt, nonce=_unb64(payload['salt'],16),_unb64(payload['nonce'],12)
    pub=X25519PublicKey.from_public_bytes(_unb64(payload['ephemeral_public_key'],32))
    shared=_private(token).exchange(pub)
    key=HKDF(algorithm=hashes.SHA256(),length=32,salt=salt,
             info=CONTEXT+b'/image').derive(shared)
    aad=MAGIC.encode()+b'|'+digest.encode()+b'|'+output_public
    raw=AESGCM(key).decrypt(nonce,_unb64(payload['ciphertext']),aad)
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Photo SHA mismatch')
    kind='jpeg' if raw.startswith(bytes.fromhex('ffd8ff')) else 'png' if raw.startswith(
        bytes.fromhex('89504e470d0a1a0a')) else 'unknown'
    if kind!=payload['source_format']:
        raise ValueError('Photo format mismatch')
    return raw

def main() -> None:
    action=sys.argv[1] if len(sys.argv)>1 else None
    if action=='public-key' and len(sys.argv)==2:
        # Public key is not sensitive, unlike the Kaggle secret.
        print(public_key(os.environ.get('KAGGLE_API_TOKEN','')))
    elif action=='decrypt' and len(sys.argv)==4:
        payload=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
        raw=decrypt(payload,os.environ.get('KAGGLE_API_TOKEN',''))
        dest=Path(sys.argv[3])
        dest.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
        with dest.open('xb') as f: f.write(raw)
        os.chmod(dest,0o600)
        print('FREEVIDEO_PRIVATE_REFERENCE_DECRYPTED_SHA256_VERIFIED')
    else:
        raise SystemExit('Usage: public-key OR decrypt ENCRYPTED_JSON PRIVATE_OUTPUT')
if __name__=='__main__':
    main()
