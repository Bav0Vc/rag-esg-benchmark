import os
import sys
from datetime import datetime


class _TimestampedWriter:
  def __init__(self, stream):
    self._stream = stream
    self._at_line_start = True

  def write(self, data):
    if not data:
      return
    result = []
    for char in data:
      if self._at_line_start and char not in ('\r', '\n'):
        result.append(datetime.now().strftime("[%d-%m-%Y %H:%M:%S] "))
        self._at_line_start = False
      result.append(char)
      if char == '\n':
        self._at_line_start = True
    self._stream.write(''.join(result))

  def flush(self):
    self._stream.flush()

  def fileno(self):
    return self._stream.fileno()

  def isatty(self):
    return False


class _Tee:
  def __init__(self, raw_stream, timestamped_log):
    self._raw = raw_stream
    self._log = timestamped_log

  def write(self, data):
    self._raw.write(data)
    self._log.write(data)

  def flush(self):
    self._raw.flush()
    self._log.flush()

  def fileno(self):
    return self._raw.fileno()

  def isatty(self):
    return self._raw.isatty()


def setup_logging(script_name: str) -> None:
  logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
  os.makedirs(logs_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
  log_path = os.path.join(logs_dir, f"{script_name}_{timestamp}.log")
  log_file = open(log_path, "w", encoding="utf-8", buffering=1)
  timestamped = _TimestampedWriter(log_file)
  sys.stdout = _Tee(sys.__stdout__, timestamped)
  sys.stderr = _Tee(sys.__stderr__, timestamped)
  print(f"Logging to {log_path}")
