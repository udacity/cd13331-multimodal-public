# Fixes for HF datasets on a read-only file system
import filelock
import contextlib

# Create a proper no-op FileLock class
class NoOpFileLock:
    def __init__(self, lock_file, *args, **kwargs):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args, **kwargs):
        pass

# Replace FileLock with a no-op context manager
filelock.FileLock = NoOpFileLock

