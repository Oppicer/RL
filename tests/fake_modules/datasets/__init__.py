"""Fake datasets module for tests"""

def load_dataset(*args, **kwargs):
    class Dummy:
        def take(self, n):
            for i in range(n):
                yield {"content": f"sample {i}"}
    return Dummy()
