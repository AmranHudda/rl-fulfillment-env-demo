"""MCP tools backed exclusively by a remote JSON-lines controller."""
import argparse
import json
import socket
import threading

from mcp.server.fastmcp import FastMCP


class ControllerClient:
    def __init__(self, address):
        self.address = address
        self.stream = self.connection = None
        self.description = None
        self.lock = threading.Lock()

    def close(self):
        if self.stream:
            self.stream.close()
        if self.connection:
            self.connection.close()
        self.stream = self.connection = None

    def read(self):
        line = self.stream.readline(65537)
        if not line or len(line) > 65536:
            raise ConnectionError("Controller disconnected or response too large")
        return line.decode().rstrip("\n")

    def call(self, tool=None, args=None, request_id=None):
        with self.lock:
            try:
                if self.stream is None:
                    self.connection = socket.create_connection(self.address, timeout=60)
                    self.stream = self.connection.makefile("rwb")
                    self.description = self.read()
                if tool is None:
                    return self.description
                request = {"tool": tool, "args": args or {}}
                if request_id is not None:
                    request["request_id"] = request_id
                self.stream.write((json.dumps(request) + "\n").encode())
                self.stream.flush()
                response = self.read()
                result = json.loads(response)
                if result.get("terminated") or result.get("truncated"):
                    self.close()
                return response
            except Exception:
                self.close()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", default="127.0.0.1:7000")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    host, port = args.controller.rsplit(":", 1)
    client = ControllerClient((host, int(port)))
    mcp = FastMCP("Fulfillment controller", host=args.host, port=args.port)

    @mcp.tool()
    def describe_task() -> str:
        return client.call()

    @mcp.tool()
    def list_tickets() -> str:
        return client.call("list_tickets")

    @mcp.tool()
    def inventory() -> str:
        return client.call("inventory")

    @mcp.tool()
    def get_order(order_id: str) -> str:
        return client.call("get_order", {"order_id": order_id})

    @mcp.tool()
    def reserve(order_id: str, warehouse: str, request_id: str) -> str:
        return client.call("reserve", {"order_id": order_id, "warehouse": warehouse}, request_id)

    @mcp.tool()
    def ship(order_id: str, request_id: str) -> str:
        return client.call("ship", {"order_id": order_id}, request_id)

    @mcp.tool()
    def close_ticket(ticket_id: str, receipt: str, request_id: str) -> str:
        return client.call("close_ticket", {"ticket_id": ticket_id, "receipt": receipt}, request_id)

    @mcp.tool()
    def finish() -> str:
        return client.call("finish")

    try:
        mcp.run(transport="streamable-http" if args.transport == "http" else "stdio")
    finally:
        client.close()


if __name__ == "__main__":
    main()
