import hashlib

def mkhash(string: str):
    m = hashlib.sha3_512()
    barray = string.encode("utf-8")
    m.update(barray)
    digest = m.digest()
    return int.from_bytes(digest)