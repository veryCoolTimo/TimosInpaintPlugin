import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Логи и отладочные файлы тестов — во временную папку, а не в настоящий
# $TMPDIR/ae-inpaint/server.log пользователя. Должно стоять до import main.
_tmp = tempfile.mkdtemp(prefix="ae-inpaint-tests-")
atexit.register(shutil.rmtree, _tmp, ignore_errors=True)
os.environ["TMPDIR"] = _tmp
tempfile.tempdir = _tmp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
