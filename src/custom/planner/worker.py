"""One owned CPU model process per planner host, with bounded request waits."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import threading
import time


class LocalWorker:
    def __init__(self, helper=None):
        self.helper = Path(helper) if helper else Path(__file__).with_name('local_model.py')
        self.process = None
        self._stderr = None
        self._key = None
        self._lock = threading.Lock()

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            process.stdin.close()
            process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def generate(self, payload, timeout=45.):
        began = time.monotonic()
        if not self._lock.acquire(timeout=timeout):
            raise subprocess.TimeoutExpired(str(self.helper), timeout)
        try:
            key = (sys.executable, os.environ.get('HAIC_LOCAL_PLANNER_MODEL'))
            if self.process is None or self.process.poll() is not None or key != self._key:
                self.close()
                self._stderr = tempfile.TemporaryFile()
                self.process = subprocess.Popen([sys.executable, '-u', str(self.helper), '--worker'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1'})
                self._key = key
            encoded = json.dumps(payload, ensure_ascii=True).encode() + b'\n'
            if len(encoded) > 32000:
                raise ValueError('planner request is too large')
            # Nonblocking I/O keeps the same deadline on a stuck write, model
            # load, inference, EOF, or a partial/non-newline worker response.
            deadline = began + timeout
            incoming = bytearray()
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                while b'\n' not in incoming:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(str(self.helper), timeout)
                    for ready, _ in selector.select(timeout=remaining):
                        if ready.fileobj is self.process.stdin:
                            count = os.write(self.process.stdin.fileno(), encoded)
                            encoded = encoded[count:]
                            if not encoded:
                                selector.unregister(self.process.stdin)
                        else:
                            block = os.read(self.process.stdout.fileno(), 4096)
                            if not block:
                                self._stderr.seek(max(0, self._stderr.tell() - 1000))
                                detail = self._stderr.read().decode(errors='replace')
                                raise RuntimeError('local planner worker exited: ' + detail)
                            incoming.extend(block)
                            if len(incoming) > 64000:
                                raise ValueError('planner worker response is too large')
            response = json.loads(incoming)
            if set(response) != {'answer'} or not isinstance(response.get('answer'), str):
                raise RuntimeError('local planner LLM failed: ' + str(response.get('error', 'invalid worker reply')))
            return response['answer']
        except BaseException:
            self.close()
            raise
        finally:
            self._lock.release()


_WORKER = LocalWorker()
atexit.register(_WORKER.close)


def generate_local(payload, timeout=45.):
    return _WORKER.generate(payload, timeout=timeout)
