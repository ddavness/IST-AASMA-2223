import hashlib

def mkhash(string: str):
    m = hashlib.sha3_512()
    barray = string.encode("utf-8")
    m.update(barray)
    digest = m.digest()
    print((int.from_bytes(digest, byteorder="big") % 708000) / 708000)
    return (int.from_bytes(digest, byteorder="big") % 708000) / 708000