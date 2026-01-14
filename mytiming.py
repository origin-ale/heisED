import time

def perf_counter():
  """
  Just a wrapper for `time.perf_counter()`
  """
  return time.perf_counter()

def timeprint(start_time, string):
  """
  Print string preceded by time elapsed since start
  """
  print(f'{time.perf_counter() - start_time:.3f}: ' + string)
  return