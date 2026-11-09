"""Authenticated private-art return for a public technical build runner.

The runner receives only an RSA PUBLIC key. Raw drawings, the AES content key
and the recipient's private key are never uploaded as public build artifacts.
This is a transport envelope, not an account credential or publication gate.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

MAGIC=b"TWSEALED1\n"
MAX_BYTES=150_000_000


def generate_keys(private:Path,public:Path)->None:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    if private.exists() or public.exists():
        raise FileExistsError("Preserve existing recipient keys")
    key=rsa.generate_private_key(public_exponent=65537,key_size=3072)
    private.parent.mkdir(parents=True,exist_ok=True)
    public.parent.mkdir(parents=True,exist_ok=True)
    # Create with private permissions from the first write.
    descriptor=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,"wb") as stream:
        stream.write(key.private_bytes(serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    public.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                       serialization.PublicFormat.SubjectPublicKeyInfo))


def seal(source:Path,public:Path,output:Path,context:dict)->dict:
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import padding,rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if output.exists():
        raise FileExistsError("Do not overwrite a private-art envelope")
    if not source.is_file() or not 0<source.stat().st_size<=MAX_BYTES:
        raise ValueError("Private artifact missing or too large")
    key=serialization.load_pem_public_key(public.read_bytes())
    if not isinstance(key,rsa.RSAPublicKey) or key.key_size<3072:
        raise ValueError("Use an RSA recipient public key with at least 3072 bits")
    content_key=AESGCM.generate_key(bit_length=256)
    nonce=os.urandom(12)
    wrapped=key.encrypt(content_key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),
                                               algorithm=hashes.SHA256(),label=None))
    header={"schema":1,"context":context,"nonce":base64.b64encode(nonce).decode(),
            "wrapped_key":base64.b64encode(wrapped).decode(),
            "recipient_sha256":hashlib.sha256(public.read_bytes()).hexdigest()}
    encoded=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    ciphertext=AESGCM(content_key).encrypt(nonce,source.read_bytes(),encoded)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream:
        stream.write(MAGIC+len(encoded).to_bytes(4,"big")+encoded+ciphertext)
    return {"sealed_file":output.name,"bytes":output.stat().st_size,
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"raw_art_uploaded":False}


def unseal(source:Path,private:Path,output:Path,expected_context:dict)->dict:
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import padding,rsa
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if output.exists():
        raise FileExistsError("Do not overwrite previously returned private artwork")
    if source.stat().st_size>MAX_BYTES+20_000:
        raise ValueError("Private envelope too large")
    data=source.read_bytes()
    if not data.startswith(MAGIC) or len(data)<len(MAGIC)+4:
        raise ValueError("Not a Studio sealed-artifact envelope")
    n=int.from_bytes(data[len(MAGIC):len(MAGIC)+4],"big")
    if not 1<=n<=16_000:
        raise ValueError("Invalid envelope header size")
    encoded=data[len(MAGIC)+4:len(MAGIC)+4+n]
    header=json.loads(encoded)
    if header.get("schema")!=1 or header.get("context")!=expected_context:
        raise ValueError("Returned art belongs to another generation request/source")
    key=serialization.load_pem_private_key(private.read_bytes(),password=None)
    if not isinstance(key,rsa.RSAPrivateKey):
        raise ValueError("Wrong private-art recipient key type")
    content_key=key.decrypt(base64.b64decode(header["wrapped_key"],validate=True),
                padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    plain=AESGCM(content_key).decrypt(base64.b64decode(header["nonce"],validate=True),
                                   data[len(MAGIC)+4+n:],encoded)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("xb") as stream:
        stream.write(plain)
    return {"file":str(output),"sha256":hashlib.sha256(plain).hexdigest(),"authenticated":True}


def main():
    parser=argparse.ArgumentParser(description="Encrypt/decrypt private Studio drawing outputs")
    sub=parser.add_subparsers(dest="command",required=True)
    gen=sub.add_parser("keys")
    gen.add_argument("--private",type=Path,required=True)
    gen.add_argument("--public",type=Path,required=True)
    for mode in ("seal","open"):
        cmd=sub.add_parser(mode)
        cmd.add_argument("--source",type=Path,required=True)
        cmd.add_argument("--key",type=Path,required=True)
        cmd.add_argument("--output",type=Path,required=True)
        cmd.add_argument("--context",type=Path,required=True)
    a=parser.parse_args()
    if a.command=="keys":
        generate_keys(a.private,a.public)
        result={"public":str(a.public),"private_key_printed":False}
    else:
        context=json.loads(a.context.read_text())
        result=(seal if a.command=="seal" else unseal)(a.source,a.key,a.output,context)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
