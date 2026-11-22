# Fulfillment environment prototype

A small offline, in-memory tool environment for experiments with state-based
task verification. All orders, tickets and inventory are fictional.

This is a vertical slice, not a production RL platform, an OS sandbox, a trained
policy or evidence of model performance on long-horizon tasks.

## Task

Fulfill the tickets that do not require manual review. Read the order and stock,
reserve inventory, create a shipment and close the ticket with its real receipt.
Leave review-only tickets and unrelated orders unchanged.

The controller awards terminal reward only when eligible orders are shipped,
their tickets are closed with matching receipts, stock reconciles, and protected
state remains unchanged. An actor's claim of success is not a grading input.

## Run

The environment and TCP controller use only the Python standard library. No
credentials, external services or model calls are needed. All task data is offline
and fictional; the servers below use local networking.

```sh
python3 -m unittest -v
python3 demo.py
python3 environment.py 7
```

The last command is a JSON-lines interface. Its first response describes the task
and tools. Send requests such as:

```json
{"tool":"list_tickets","args":{}}
```

Mutations require a `request_id`. Once a well-formed mutation is attempted,
its ID is bound to the payload and its result, including a rejection by the
tool. Reusing the ID with the same request returns that result without
repeating the mutation; using it with another payload is rejected. To retry a
rejected mutation after the world changes, use a new ID. `finish` ends the
episode and returns the state-based reward.

## Isolated controller (JSON-lines TCP)

```sh
python3 controller_server.py --seed 7 --port 7000 --log episodes.jsonl
# Alternatively: --seeds 7-10 cycles operator-selected seeds per connection.
# Evaluation without retries: --seeds 7-10 --each-seed-once --log evaluation.jsonl
```

The default bind is `127.0.0.1`; use `--host` explicitly to expose it elsewhere.
Each TCP connection gets a fresh, independent episode in a controller thread.
The controller sends the reset description first, then accepts the same JSON
request lines as the stdin interface and sends exactly its response envelopes.
It closes the connection after termination/truncation. Invalid JSON returns
`{"error":"Invalid JSON","reward":0}`. Lines over 64 KiB are rejected and the
connection closes; idle connections close after 60 seconds (`--idle-timeout`).
Non-standard JSON constants (`NaN`, `Infinity`) and numeric float overflow are
also rejected as invalid JSON, keeping controller logs strict JSON.
Only the operator chooses seeds; there is no seed/reset tool. Optional JSONL
logs contain seed, attempt, steps, reward and final snapshot, stay on the controller side,
and must not be mounted or otherwise made accessible to an actor.

`--each-seed-once` works with either `--seed` or `--seeds`: each seed is played at
most once per controller run. A connection reserves a seed before the initial
description, and consumes it when its first request line arrives, including
invalid JSON or an oversized line. Concurrent once-mode connections cannot
reserve the same seed. If all seeds are consumed or currently reserved, a new
connection gets `{"error":"No seeds remaining","reward":0,"terminated":true,
"truncated":false}` and closes without a log record. An unused reservation goes
back to the front of the pool when its connection closes or idles out.

In both modes, connections that send no request do not advance the seed cycle,
count as attempts or create log entries; readiness probes therefore do not play
episodes. Default mode still cycles and permits repeated attempts. Each played
episode has a 1-based `attempt` counter for its seed (always 1 in once-mode),
assigned on its first request; records are appended when episodes end, so
concurrently played episodes can finish in a different order. Counters and seed
consumption are in-memory per controller run, not recovered from an existing
append-only log: use a fresh log for each evaluation run.

## MCP tools

```sh
python3 -m pip install -r requirements-server.txt
python3 mcp_server.py --controller 127.0.0.1:7000
```

The official MCP SDK is pinned in `requirements-server.txt`; it is not needed to
import or run `environment.py` or the TCP controller. The bridge never imports
the environment or owns its state. It connects lazily and forwards controller
JSON unchanged as MCP text tool results. `describe_task` returns the initial
description. The other tools are `list_tickets`, `inventory`, `get_order`,
`reserve`, `ship`, `close_ticket` and `finish` with the same arguments (mutations
also require `request_id`). After a terminal result the next call starts a new
episode. One bridge instance owns one episode connection shared by its callers;
use separate bridge processes for independent actors.
If an established controller connection is lost (including idle timeout), the
detecting call returns `{"error":"Controller connection lost; episode ended",
"reward":0,"terminated":true,"truncated":false}` rather than silently retrying
an action in a new episode. The bridge clears its cached description; the next
call connects to a fresh episode. `describe_task` also checks for disconnection
before returning a cached description. Initial connection failures remain MCP
tool errors rather than claiming an episode was started.
An initial controller error such as `No seeds remaining` is returned unchanged
as the tool result, not mistaken for a task description.

A coding agent's MCP configuration can use stdio (replace the absolute path):

```json
{"mcpServers":{"fulfillment":{"command":"python3","args":["/absolute/path/mcp_server.py","--controller","127.0.0.1:7000"]}}}
```

Or run `python3 mcp_server.py --controller 127.0.0.1:7000 --transport http --port 8000`
and add the streamable HTTP URL `http://127.0.0.1:8000/mcp` to the agent's MCP
configuration (for clients supporting URL configuration:
`{"mcpServers":{"fulfillment":{"url":"http://127.0.0.1:8000/mcp"}}}`).

## Docker

```sh
docker build -t fulfillment-controller .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
  --tmpfs /tmp -p 127.0.0.1:7000:7000 -p 127.0.0.1:8000:8000 fulfillment-controller --seed 7
```

The slim Python image runs as a non-root user and supervises both the controller
on 7000 and the MCP HTTP bridge on 8000, bound to `0.0.0.0` inside the container.
Give the actor only the TCP port or MCP URL, not a shell, Docker socket, shared
filesystem mounts or controller logs. The command publishes only on loopback.
Publishing more widely lets any network peer act on and end the shared MCP
episode: there is no authentication. Container arguments are controller options;
keep its port at 7000 because the bridge uses that internal port.
The recommended hardening flags make the root filesystem read-only, drop Linux
capabilities and prevent privilege escalation; `/tmp` remains writable in memory
for runtime temporary files. Integration tests use these same flags and solve
through both published ports, including the SDK's streamable HTTP client on 8000.
If logging is enabled with `--log`, provision a separate controller-only writable
volume for that path; do not expose it to the actor.
The entrypoint stops both servers if either exits, and exits nonzero when a child
fails on its own so startup failures are visible to container tooling. Requested
SIGTERM/SIGINT shutdown exits zero; forwarded shutdown signals are not failures.

## Evidence and limits

- 57 tests: the original 19 unchanged tests (including scripted solutions across
  100 seeds and immutable trace snapshots), 19 grader-audit tests, and nineteen
  isolation integration tests covering
  real subprocess/socket episodes, protocol probes, concurrent independence,
  operator seeds, private logs, defensive input/idle handling, SDK stdio and
  streamable HTTP MCP, explicit controller disconnects and broken-write recovery,
  and hardened Docker TCP
  and MCP HTTP published-port solves, strict JSON logs and entrypoint failures.
  MCP tests (including the Docker MCP subtest) skip without the SDK; Docker skips
  without a working daemon. Run `python3 -m unittest -v` for counts and skips.
- The score is every episode in the log, never the best one. Include failed,
  abandoned and invalid-request episodes, not just reward-1 episodes. Use
  `--each-seed-once` and a fresh controller/log for a bounded no-retry evaluation;
  cycling mode is for development and records every played attempt. Tests check
  exact episode counts, seeds, attempts and rewards, including readiness probes,
  exhaustion and concurrent once-mode reservations.
- Different valid action orders are accepted.
- Negative cases cover fabricated receipts, missing reservations, protected
  orders, malformed requests, mutation replay, reset, exhausted step budgets and
  false completion claims.
- The [grader audit](GRADER_AUDIT.md) found no false positive or protected-state
  change in 1,536 sampled episodes. It fixed rejected-mutation replay and a
  controller-side error on a missing order. Tests detected 65 of 67 single
  condition mutants; the other two are redundant with intact checks. This is
  not exhaustive and does not verify whether the actor reads order or stock
  before acting: the reward checks the final state, not the action history.
  That audit did not exercise process isolation; the separate isolation tests
  cover the port-only process boundary described below, not direct host access.
- `demo.py` prints tool transcripts from a scripted baseline and a negative
  control. These are explicitly not LLM or trained-policy rollouts.
- With a port-only actor, the process boundary (and optional container boundary)
  provides no Python object, memory or filesystem access through the protocol:
  actors see only tool responses, cannot choose seeds, read controller logs or
  invoke controller internals. Tests demonstrate these API properties, not a
  proof against every possible attack. Reward still comes only from `grade()`
  over final controller state, never actor claims.
- This is not an actor-side sandbox. An actor with host shell permissions may
  read same-user files/processes; an actor with host Docker access could use
  `docker exec`. Do not grant those capabilities when relying on isolation.
- There is no authentication or TLS: anyone reaching a port can act. There is
  no hardening against container escape or kernel exploits, and no rate limiting
  or resource quotas beyond line-size and idle-timeout checks (connections and
  logs can consume resources). The MCP layer relies on the SDK. Production
  persistence and a hardened multi-tenant service are not implemented.
- Seeds vary identifiers and stock counts; they do not yet establish broad task
  diversity or a held-out difficulty benchmark.
- No real customer data, Fitscout code, private recordings, photos or credentials
  are included.
