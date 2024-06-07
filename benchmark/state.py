import pickle
import threading
import time

# A thread-safe class to handle dictionary persistence
class BenchmarkState:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}
        self.lock = threading.Lock()
        self.load()

    def load(self):
        with self.lock:
            try:
                with open(self.file_path, 'rb') as file:
                    self.data = pickle.load(file)
            except FileNotFoundError:
                self.data = {}

    def save(self):
        with self.lock:
            with open(self.file_path, 'wb') as file:
                pickle.dump(self.data, file)

    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            threading.Thread(target=self.save).start()

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

    def delete(self, key):
        with self.lock:
            if key in self.data:
                del self.data[key]
                threading.Thread(target=self.save).start()

