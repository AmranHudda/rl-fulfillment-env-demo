"""Actors below use only wire responses, never controller Python objects."""
import asyncio
import contextlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
TOOLS = {"describe_task", "list_tickets", "inventory", "get_order", "reserve",
         "ship", "close_ticket", "finish"}


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Actor:
    def __init__(self, port):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.stream = self.socket.makefile("rwb")
        self.received = []
        try:
            self.description = self.read()
        except Exception:
            self.close()
            raise

    def read(self):
        line = self.stream.readline()
        self.received.append(line.decode())
        return json.loads(line)

    def raw(self, line):
        self.stream.write(line)
        self.stream.flush()
        return self.read()

    def request(self, request):
        return self.raw((json.dumps(request) + "\n").encode())

    def call(self, tool, args=None, key=None):
        request = {"tool": tool, "args": args or {}}
        if key is not None:
            request["request_id"] = key
        return self.request(request)

    def close(self):
        self.stream.close()
        self.socket.close()


async def solve(call):
    tickets = (await call("list_tickets"))["observation"]
    stock = (await call("inventory"))["observation"]
    for tid, ticket in tickets.items():
        oid = ticket["order_id"]
        order = (await call("get_order", {"order_id": oid}))["observation"]
        if order["needs_review"]:
            continue
        warehouse = next(w for w, values in stock.items()
                         if values.get(order["sku"], 0) >= order["qty"])
        result = await call("reserve", {"order_id": oid, "warehouse": warehouse}, tid + "r")
        assert "error" not in result, result
        stock[warehouse][order["sku"]] -= order["qty"]
        receipt = (await call("ship", {"order_id": oid}, tid + "s"))["observation"]["receipt"]
        result = await call("close_ticket", {"ticket_id": tid, "receipt": receipt}, tid + "c")
        assert "error" not in result, result
    return await call("finish")


def socket_solve(actor):
    async def call(*args):
        return actor.call(*args)
    return asyncio.run(solve(call))


async def http_solve(test, port):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client("http://127.0.0.1:%s/mcp" % port) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            test.assertEqual({tool.name for tool in (await session.list_tools()).tools}, TOOLS)

            async def call(tool, args=None, key=None):
                arguments = dict(args or {})
                if key is not None:
                    arguments["request_id"] = key
                result = await session.call_tool(tool, arguments)
                test.assertFalse(result.isError, result)
                return json.loads(result.content[0].text)

            test.assertEqual(set(await call("describe_task")), {"task", "tools", "max_steps"})
            test.assertEqual((await solve(call))["reward"], 1)
            test.assertEqual((await call("finish"))["reward"], 0)


def wait_http_ready(port, running):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not running():
            raise RuntimeError("HTTP server exited")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("HTTP server did not listen")


def wait_ready(port, process):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited: %s" % process.returncode)
        try:
            actor = Actor(port)
            actor.socket.shutdown(socket.SHUT_WR)
            actor.stream.readline()
            actor.close()
            return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError("Server did not listen")


@contextlib.contextmanager
def controller(*options):
    port = free_port()
    process = subprocess.Popen([sys.executable, str(ROOT / "controller_server.py"),
                                "--port", str(port), *options], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        wait_ready(port, process)
        yield port
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.context = controller("--seed", "7", "--idle-timeout", "1")
        self.port = self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    def actor(self):
        actor = Actor(self.port)
        self.addCleanup(actor.close)
        return actor

    def test_socket_solve_and_empty_finish(self):
        actor = self.actor()
        self.assertEqual(set(actor.description), {"task", "tools", "max_steps"})
        self.assertEqual(socket_solve(actor), {"observation": {"complete": True},
                         "reward": 1, "terminated": True, "truncated": False})
        self.assertEqual(actor.stream.readline(), b"")
        self.assertEqual(self.actor().call("finish")["reward"], 0)

    def test_probes_cannot_change_state_or_seed(self):
        actor, control = self.actor(), self.actor()
        tickets = actor.call("list_tickets")
        inventory = actor.call("inventory")
        oid = next(iter(tickets["observation"].values()))["order_id"]
        before = actor.call("get_order", {"order_id": oid})
        for tool in ("snapshot", "grade", "_state", "set_reward", "reset", "seed"):
            result = actor.call(tool)
            self.assertEqual(set(result), {"error", "reward", "terminated", "truncated"})
            self.assertEqual(result["reward"], 0)
        for field in ("snapshot", "grade", "_state", "set_reward", "reset", "seed"):
            self.assertIn("error", actor.request({"tool": "inventory", field: 999}))
        self.assertIn("error", actor.call("inventory", {"seed": "999"}))
        self.assertEqual(actor.call("list_tickets"), tickets)
        self.assertEqual(actor.call("inventory"), inventory)
        self.assertEqual(actor.call("get_order", {"order_id": oid}), before)
        self.assertEqual(control.call("list_tickets"), tickets)
        self.assertEqual(set(inventory), {"observation", "reward", "terminated", "truncated"})
        self.assertEqual(socket_solve(actor)["reward"], 1)

    def test_concurrent_connections_are_independent(self):
        first, second = self.actor(), self.actor()
        tickets = first.call("list_tickets")["observation"]
        oid = next(iter(tickets.values()))["order_id"]
        initial = second.call("inventory")
        first.call("reserve", {"order_id": oid, "warehouse": "north"}, "reserve")
        self.assertNotEqual(first.call("inventory"), initial)
        self.assertEqual(second.call("inventory"), initial)
        self.assertEqual(second.call("get_order", {"order_id": oid})["observation"]["status"], "open")
        self.assertEqual(first.call("get_order", {"order_id": oid})["observation"]["status"], "reserved")

    def test_garbage_oversized_and_idle(self):
        actor = self.actor()
        for line in (b"garbage\n", b"\xff\n", b"\n"):
            self.assertEqual(actor.raw(line), {"error": "Invalid JSON", "reward": 0})
        self.assertIn("observation", actor.call("inventory"))
        self.assertEqual(actor.raw(b"x" * 65537 + b"\n"),
                         {"error": "Line too long", "reward": 0})
        self.assertEqual(actor.stream.readline(), b"")
        idle = self.actor()
        self.assertEqual(idle.stream.readline(), b"")
        self.assertEqual(socket_solve(self.actor())["reward"], 1)

    def test_operator_seed_cycle(self):
        with controller("--seeds", "7-8") as port:
            actors = [Actor(port) for _ in range(3)]
            try:
                tickets = [a.call("list_tickets") for a in actors]
                self.assertEqual(tickets[0], tickets[2])
                self.assertNotEqual(tickets[0], tickets[1])
            finally:
                for actor in actors:
                    actor.close()

    def test_log_is_controller_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-log.jsonl"
            with controller("--seed", "7", "--log", str(path)) as port:
                actor = Actor(port)
                try:
                    self.assertEqual(socket_solve(actor)["reward"], 1)
                    self.assertEqual(actor.stream.readline(), b"")
                    wire = "".join(actor.received)
                finally:
                    actor.close()
                empty = Actor(port)
                try:
                    self.assertEqual(empty.call("finish")["reward"], 0)
                    self.assertEqual(empty.stream.readline(), b"")
                    wire += "".join(empty.received)
                finally:
                    empty.close()
            logs = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(log["seed"], log["attempt"], log["reward"]) for log in logs],
                             [(7, 1, 1), (7, 2, 0)])
            episode = logs[0]
            self.assertEqual(set(episode), {"seed", "attempt", "steps", "reward", "final_snapshot"})
            hidden = set(episode["final_snapshot"]["orders"]) - {
                t["order_id"] for t in episode["final_snapshot"]["tickets"].values()}
            self.assertTrue(hidden)
            for value in [str(path), "final_snapshot", '"seed"', *hidden]:
                self.assertNotIn(value, wire)

    def test_nonfinite_json_rejected_and_log_is_strict(self):
        def reject(value):
            raise ValueError("Non-finite JSON")

        def finite(value):
            from decimal import Decimal
            number = Decimal(value)
            if not number.is_finite():
                raise ValueError("Non-finite JSON")
            return number

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "strict.jsonl"
            with controller("--log", str(path)) as port:
                actor = Actor(port)
                try:
                    for value in (b"NaN", b"Infinity", b"-Infinity", b"1e400", b"-1e400"):
                        line = b'{"tool":"inventory","args":{},"request_id":' + value + b'}\n'
                        self.assertEqual(actor.raw(line), {"error": "Invalid JSON", "reward": 0})
                    self.assertIn("observation", actor.call("inventory"))
                    actor.call("finish")
                    self.assertEqual(actor.stream.readline(), b"")
                finally:
                    actor.close()
            logs = [json.loads(line, parse_constant=reject, parse_float=finite)
                    for line in path.read_text().splitlines()]
            self.assertEqual(len(logs), 1)
            episode = logs[0]
            self.assertEqual(len(episode["steps"]), 7)
            self.assertEqual([step["request"]["tool"] for step in episode["steps"]
                              if step["request"] is not None],
                             ["inventory", "finish"])


class SeedAllocationTests(unittest.TestCase):
    def refusal(self, port):
        actor = Actor(port)
        try:
            self.assertEqual(actor.description, {"error": "No seeds remaining", "reward": 0,
                                                "terminated": True, "truncated": False})
            self.assertEqual(actor.stream.readline(), b"")
        finally:
            actor.close()

    def finish(self, port):
        actor = Actor(port)
        try:
            self.assertIn("task", actor.description)
            self.assertEqual(actor.call("finish")["reward"], 0)
            self.assertEqual(actor.stream.readline(), b"")
        finally:
            actor.close()

    def test_once_mode_consumes_each_seed_including_invalid_requests(self):
        for seeds in (("--seeds", "7-8"), ("--seed", "7")):
            with self.subTest(seeds=seeds), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episodes.jsonl"
                with controller(*seeds, "--each-seed-once", "--log", str(path)) as port:
                    actor = Actor(port)
                    try:
                        self.assertEqual(actor.raw(b"garbage\n"),
                                         {"error": "Invalid JSON", "reward": 0})
                        actor.socket.shutdown(socket.SHUT_WR)
                        self.assertEqual(actor.stream.readline(), b"")
                    finally:
                        actor.close()
                    if seeds[0] == "--seeds":
                        self.finish(port)
                    self.refusal(port)
                logs = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([(log["seed"], log["attempt"], log["reward"]) for log in logs],
                                 [(7, 1, 0), (8, 1, 0)] if seeds[0] == "--seeds" else [(7, 1, 0)])

    def test_no_request_probes_release_without_log_or_cycle_advance(self):
        for options in ((), ("--each-seed-once",)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episodes.jsonl"
                with controller("--seeds", "7-8", "--idle-timeout", "0.2", "--log", str(path),
                                *options) as port:
                    for idle in (False, True):
                        probe = Actor(port)
                        try:
                            if not idle:
                                probe.socket.shutdown(socket.SHUT_WR)
                            self.assertEqual(probe.stream.readline(), b"")
                        finally:
                            probe.close()
                    self.assertEqual(path.read_text(), "")
                    self.finish(port)
                    self.finish(port)
                logs = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([(log["seed"], log["attempt"]) for log in logs], [(7, 1), (8, 1)])

    def test_concurrent_once_mode_reservations_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            with controller("--seeds", "7-8", "--each-seed-once", "--log", str(path)) as port:
                first, second = Actor(port), Actor(port)
                try:
                    self.refusal(port)
                    self.assertNotEqual(first.call("list_tickets"), second.call("list_tickets"))
                    for actor in (first, second):
                        actor.call("finish")
                        self.assertEqual(actor.stream.readline(), b"")
                    self.refusal(port)
                finally:
                    first.close()
                    second.close()
            logs = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(log["seed"], log["attempt"]) for log in logs], [(7, 1), (8, 1)])

    def test_cycling_attempt_numbers_are_per_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            with controller("--seeds", "7-8", "--log", str(path)) as port:
                for _ in range(4):
                    self.finish(port)
            logs = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(log["seed"], log["attempt"]) for log in logs],
                             [(7, 1), (8, 1), (7, 2), (8, 2)])


@unittest.skipUnless(importlib.util.find_spec("mcp"), "mcp SDK not installed")
class MCPTests(unittest.TestCase):
    def test_exhaustion_first_line_is_tool_result(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def run(port):
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / "mcp_server.py"), "--controller", "127.0.0.1:%s" % port])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("finish", {})
                    self.assertFalse(result.isError)
                    self.assertTrue(json.loads(result.content[0].text)["terminated"])
                    for tool in ("describe_task", "inventory"):
                        result = await session.call_tool(tool, {})
                        self.assertFalse(result.isError)
                        self.assertEqual(json.loads(result.content[0].text), {
                            "error": "No seeds remaining", "reward": 0,
                            "terminated": True, "truncated": False})

        with controller("--seed", "7", "--each-seed-once") as port:
            asyncio.run(run(port))

    def test_broken_connection_write_cleanup_and_reconnect(self):
        from mcp_server import ControllerClient

        with controller("--seed", "7") as port:
            client = ControllerClient(("127.0.0.1", port))
            try:
                for args in ({}, {"padding": "x" * 16384}):
                    with self.subTest(buffered=not args):
                        self.assertIn("observation", json.loads(client.call("inventory")))
                        client.connection.shutdown(socket.SHUT_RDWR)
                        self.assertEqual(json.loads(client.call("inventory", args)), {
                            "error": "Controller connection lost; episode ended", "reward": 0,
                            "terminated": True, "truncated": False})
                        self.assertIsNone(client.stream)
                        self.assertIsNone(client.connection)
                        self.assertIsNone(client.description)
                        self.assertEqual(set(json.loads(client.call())),
                                         {"task", "tools", "max_steps"})
                        self.assertIn("observation", json.loads(client.call("inventory")))
            finally:
                client.close()

    def test_controller_idle_loss_is_explicit(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def run(port):
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / "mcp_server.py"), "--controller", "127.0.0.1:%s" % port])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    async def call(tool):
                        result = await session.call_tool(tool, {})
                        self.assertFalse(result.isError, result)
                        return json.loads(result.content[0].text)

                    for tool in ("inventory", "describe_task"):
                        before = await call("list_tickets")
                        await asyncio.sleep(0.5)
                        self.assertEqual(await call(tool), {
                            "error": "Controller connection lost; episode ended", "reward": 0,
                            "terminated": True, "truncated": False})
                        self.assertEqual(set(await call("describe_task")),
                                         {"task", "tools", "max_steps"})
                        self.assertNotEqual(await call("list_tickets"), before)
                        self.assertEqual((await call("finish"))["reward"], 0)

        with controller("--seeds", "7-8", "--idle-timeout", "0.2") as port:
            asyncio.run(run(port))

    def test_sdk_http_tools_and_solve(self):
        with controller("--seed", "7") as port:
            http_port = free_port()
            process = subprocess.Popen([sys.executable, str(ROOT / "mcp_server.py"),
                "--controller", "127.0.0.1:%s" % port, "--transport", "http",
                "--port", str(http_port)], stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            try:
                wait_http_ready(http_port, lambda: process.poll() is None)
                asyncio.run(http_solve(self, http_port))
            finally:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def test_sdk_stdio_tools_and_solve(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def run(port):
            params = StdioServerParameters(command=sys.executable,
                args=[str(ROOT / "mcp_server.py"), "--controller", "127.0.0.1:%s" % port])
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.assertEqual({tool.name for tool in (await session.list_tools()).tools}, TOOLS)

                    async def call(tool, args=None, key=None):
                        arguments = dict(args or {})
                        if key is not None:
                            arguments["request_id"] = key
                        result = await session.call_tool(tool, arguments)
                        self.assertFalse(result.isError, result)
                        return json.loads(result.content[0].text)

                    self.assertEqual(set(await call("describe_task")), {"task", "tools", "max_steps"})
                    self.assertEqual((await solve(call))["reward"], 1)
                    self.assertEqual((await call("finish"))["reward"], 0)
                    error = await session.call_tool("snapshot", {})
                    self.assertTrue(error.isError)
        with controller("--seed", "7") as port:
            asyncio.run(run(port))


class EntrypointTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("mcp"), "mcp SDK not installed")
    def test_requested_shutdown_exits_zero(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=signum):
                port = free_port()
                process = subprocess.Popen([sys.executable, str(ROOT / "container_entrypoint.py"),
                    "--port", str(port)], cwd=ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                try:
                    wait_ready(port, process)
                    process.send_signal(signum)
                    self.assertEqual(process.wait(timeout=10), 0)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def test_controller_startup_failure_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = str(Path(directory) / "missing" / "episodes.jsonl")
            result = subprocess.run([sys.executable, str(ROOT / "container_entrypoint.py"),
                "--port", str(free_port()), "--log", log_path], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(log_path, result.stderr)


def docker_works():
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@unittest.skipUnless(docker_works(), "Docker daemon unavailable")
class DockerTests(unittest.TestCase):
    def test_container_published_port(self):
        tag = "fulfillment-isolation-test:%s" % time.time_ns()
        container_id = None
        try:
            subprocess.run(["docker", "build", "-t", tag, str(ROOT)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            container_id = subprocess.check_output(["docker", "run", "-d", "--rm",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--tmpfs", "/tmp", "-p", "127.0.0.1::7000", "-p", "127.0.0.1::8000",
                tag, "--seed", "7", "--log", "/tmp/episodes.jsonl"], text=True).strip()
            published = subprocess.check_output(["docker", "port", container_id, "7000"], text=True)
            port = int(published.strip().rsplit(":", 1)[1])
            deadline = time.monotonic() + 20
            while True:
                try:
                    actor = Actor(port)
                    break
                except (OSError, ValueError):
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            actor.socket.shutdown(socket.SHUT_WR)
            self.assertEqual(actor.stream.readline(), b"")
            actor.close()
            actor = Actor(port)
            try:
                self.assertEqual(socket_solve(actor)["reward"], 1)
            finally:
                actor.close()
            with self.subTest(transport="MCP HTTP"):
                if not importlib.util.find_spec("mcp"):
                    self.skipTest("mcp SDK not installed")
                published = subprocess.check_output(
                    ["docker", "port", container_id, "8000"], text=True)
                http_port = int(published.strip().rsplit(":", 1)[1])

                def running():
                    return subprocess.check_output(["docker", "inspect", "--format",
                        "{{.State.Running}}", container_id], text=True).strip() == "true"

                wait_http_ready(http_port, running)
                asyncio.run(http_solve(self, http_port))
            controller_log = subprocess.check_output(["docker", "exec", container_id,
                "python3", "-c", "from pathlib import Path; print(Path('/tmp/episodes.jsonl').read_text())"],
                text=True)
            logs = [json.loads(line) for line in controller_log.splitlines() if line]
            expected = [(7, 1, 1)]
            if importlib.util.find_spec("mcp"):
                expected += [(7, 2, 1), (7, 3, 0)]
            self.assertEqual([(log["seed"], log["attempt"], log["reward"]) for log in logs],
                             expected)
        finally:
            if container_id:
                subprocess.run(["docker", "stop", "-t", "5", container_id], check=True,
                               stdout=subprocess.PIPE)
            subprocess.run(["docker", "image", "rm", tag], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
