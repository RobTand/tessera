"""Read a private, action-bound local handoff without canonical payload I/O."""
import hashlib
import json
import os
from pathlib import Path
import stat


class LocalNativeInputs:
    def __init__(self, binding, action_key):
        self.root = Path(binding['path']).parent
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            owner = os.fstat(self.fd)
            if owner.st_uid != os.getuid() or owner.st_mode & 0o077:
                raise ValueError('local handoff directory is not private to this owner')
            raw = self._read(Path(binding['path']).name, binding, 8 << 20)
            self.bundle = json.loads(raw)
            if (self.bundle['schema'] != 'prismaquant.native_local_handoff.v1'
                    or not action_key or self.bundle['action_key'] != action_key):
                raise ValueError('local handoff belongs to a different admitted action')
            if not 0 < self.bundle['reserved_bytes'] <= self.bundle['max_bytes'] <= 40 << 30:
                raise ValueError('local handoff disk bound differs')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def _read(self, name, binding, limit):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('invalid local payload name')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_size != binding['bytes']
                    or before.st_size > limit or before.st_uid != os.getuid()):
                raise ValueError('invalid local payload')
            with os.fdopen(fd, 'rb', closefd=False) as handle:
                raw = handle.read(limit + 1)
            after = os.fstat(fd)
            fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
            if tuple(getattr(before, k) for k in fields) != tuple(getattr(after, k) for k in fields):
                raise ValueError('local payload changed under descriptor')
            if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
                raise ValueError('local payload digest differs')
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return raw
        finally:
            os.close(fd)

    def read(self, path, expected_sha256, limit):
        path = Path(path)
        if path.parent != self.root or path.name not in self.bundle['files']:
            raise ValueError('payload is outside owned local bundle')
        binding = self.bundle['files'][path.name]
        if binding['path'] != str(path) or binding['sha256'] != expected_sha256:
            raise ValueError('local payload binding differs')
        return self._read(path.name, binding, limit)
