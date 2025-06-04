"""Fake tiktoken module for tests"""

class DummyEnc:
    def encode(self, text):
        return [ord(c) for c in text]

def get_encoding(name):
    return DummyEnc()
