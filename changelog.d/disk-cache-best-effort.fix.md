A failed disk-cache write of any kind (including serialization errors) no
longer destroys the freshly computed result; the write is best-effort and the
failure is logged.
