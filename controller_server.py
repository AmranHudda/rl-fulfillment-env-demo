"""Local JSON-lines controller. The environment lives only in this process."""
import argparse
from collections import deque
import json
import math
import signal
import socket
import socketserver
import threading

from environment import FulfillmentEnv

MAX_LINE = 65536


def reject_constant(value):
    raise ValueError("Non-standard JSON constant")


def finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


class Controller(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, address, seeds, idle_timeout, log_path, each_seed_once=False):
        self.seeds = seeds
        self.each_seed_once = each_seed_once
        self.next_seed = 0
        self.released = deque()
        self.attempts = {}
        self.lock = threading.Lock()
        self.connections = set()
        self.idle_timeout = idle_timeout
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        super().__init__(address, Episode)

    def reserve_seed(self):
        with self.lock:
            if self.released:
                index = self.released.popleft()
            else:
                if self.each_seed_once and self.next_seed >= len(self.seeds):
                    return None
                index = self.next_seed
                self.next_seed += 1
            return index, self.seeds[index % len(self.seeds)]

    def consume_seed(self, seed):
        with self.lock:
            attempt = self.attempts.get(seed, 0) + 1
            self.attempts[seed] = attempt
            return attempt

    def process_request(self, request, client_address):
        with self.lock:
            self.connections.add(request)
        super().process_request(request, client_address)

    def shutdown_request(self, request):
        with self.lock:
            self.connections.discard(request)
        super().shutdown_request(request)

    def shutdown(self):
        super().shutdown()
        with self.lock:
            for connection in self.connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def server_close(self):
        super().server_close()
        with self.lock:
            if self.log:
                self.log.close()
                self.log = None


class Episode(socketserver.StreamRequestHandler):
    def handle(self):
        server = self.server
        self.connection.settimeout(server.idle_timeout)

        def send(value):
            self.wfile.write((json.dumps(value) + "\n").encode())
            self.wfile.flush()

        reservation = server.reserve_seed()
        if reservation is None:
            try:
                send({"error": "No seeds remaining", "reward": 0,
                      "terminated": True, "truncated": False})
            except OSError:
                pass
            return
        index, seed = reservation
        env = FulfillmentEnv(seed)
        steps = []
        invalid_lines = 0
        reward = 0
        attempt = None
        try:
            send(env.reset(seed))
            while True:
                line = self.rfile.readline(MAX_LINE + 1)
                if not line:
                    break
                if attempt is None:
                    attempt = server.consume_seed(seed)
                if len(line) > MAX_LINE:
                    response = {"error": "Line too long", "reward": 0}
                    steps.append({"request": None, "response": response})
                    send(response)
                    break
                try:
                    request = json.loads(line, parse_constant=reject_constant,
                                         parse_float=finite_float)
                except (ValueError, UnicodeDecodeError):
                    response = {"error": "Invalid JSON", "reward": 0}
                    invalid_lines += 1
                    send(response)
                    continue
                response = env.step(request)
                steps.append({"request": request, "response": response})
                reward = response["reward"]
                send(response)
                if response.get("terminated") or response.get("truncated"):
                    break
        except (OSError, TimeoutError):
            pass
        finally:
            with server.lock:
                if attempt is None:
                    server.released.appendleft(index)
                elif server.log:
                    server.log.write(json.dumps({"seed": seed, "attempt": attempt, "steps": steps,
                                                 "invalid_lines": invalid_lines,
                                                 "reward": reward,
                                                 "final_snapshot": env.snapshot}) + "\n")
                    server.log.flush()


def seed_range(value):
    try:
        start, end = map(int, value.split("-"))
        if end < start:
            raise ValueError
        return range(start, end + 1)
    except ValueError:
        raise argparse.ArgumentTypeError("expected ascending seeds A-B")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7000)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=0)
    seeds.add_argument("--seeds", type=seed_range)
    parser.add_argument("--each-seed-once", action="store_true")
    parser.add_argument("--idle-timeout", type=float, default=60)
    parser.add_argument("--log")
    args = parser.parse_args()
    with Controller((args.host, args.port), args.seeds or [args.seed],
                    args.idle_timeout, args.log, args.each_seed_once) as server:
        def shutdown(signum, frame):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        print("Controller listening on %s:%s" % server.server_address, flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
