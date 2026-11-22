"""Supervise both local servers and forward container shutdown."""
import signal
import subprocess
import sys
import time


def main():
    children = []
    shutdown_requested = False
    failure = 0

    def child_failure():
        return next((child.returncode for child in children
                     if child.poll() is not None and child.returncode != 0), 0)

    def stop(signum, frame):
        nonlocal shutdown_requested, failure
        if not shutdown_requested:
            failure = child_failure()
            shutdown_requested = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "controller_server.py",
                                          "--host", "0.0.0.0", *sys.argv[1:]]))
        children.append(subprocess.Popen([sys.executable, "mcp_server.py", "--transport",
                                          "http", "--host", "0.0.0.0", "--controller",
                                          "127.0.0.1:7000"]))
        while not shutdown_requested and all(child.poll() is None for child in children):
            time.sleep(0.1)
        if not shutdown_requested:
            failure = child_failure()
        return failure if failure >= 0 else 128 - failure
    finally:
        stop(None, None)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
