"""Print real tool transcripts from a SCRIPTED baseline and a negative control.

This is not a model rollout, a trained policy, or an independent agent benchmark.
"""
import copy
import json

from environment import FulfillmentEnv


def episode(early_finish=False):
    env = FulfillmentEnv(seed=7)
    transcript = []

    def tool(name, args=None):
        request = {"tool": name, "args": args or {}}
        if name in env.MUTATIONS:
            request["request_id"] = "request-%s" % len(transcript)
        result = env.step(request)
        transcript.append(copy.deepcopy({"request": request, "response": result}))
        return result

    if early_finish:
        tool("finish")
    else:
        tickets = tool("list_tickets")["observation"]
        inventory = tool("inventory")["observation"]
        for tid, ticket in tickets.items():
            oid = ticket["order_id"]
            order = tool("get_order", {"order_id": oid})["observation"]
            if order["needs_review"]:
                continue
            warehouse = next(w for w, stock in inventory.items()
                             if stock.get(order["sku"], 0) >= order["qty"])
            tool("reserve", {"order_id": oid, "warehouse": warehouse})
            inventory[warehouse][order["sku"]] -= order["qty"]
            receipt = tool("ship", {"order_id": oid})["observation"]["receipt"]
            tool("close_ticket", {"ticket_id": tid, "receipt": receipt})
        tool("finish")
    return {"kind": "negative_control" if early_finish else "scripted_baseline",
            "seed": 7, "transcript": transcript,
            "terminal_reward": transcript[-1]["response"]["reward"]}


if __name__ == "__main__":
    for control in (False, True):
        print(json.dumps(episode(control), sort_keys=True))
