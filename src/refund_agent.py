
"""Part D: database-backed refund tools and resumable bounded ReAct loop."""
from pathlib import Path
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone

import mock_llm
from src.policy_rag import PolicyRAG

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"

rag = PolicyRAG()


def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

    if not row:
        return {"error": "order not found"}

    return dict(zip(
        ["order_id", "status", "order_amount", "delivery_minutes"],
        row
    ))


def search_policy(query):
    return {"results": rag.retrieve(query, 3)}


def calculate_refund(order_amount, delivery_minutes):
    amount = (
        float(order_amount)
        if delivery_minutes is not None
        and float(delivery_minutes) > 45
        else 0.0
    )
    return {"refund_inr": round(amount, 2)}


def _ledger(tool, order_id, refund_inr):
    OUT.mkdir(exist_ok=True)
    path = OUT / "refund_ledger.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": refund_inr
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {"refund_inr": refund_inr}

def _validate_refund(order_id, refund_inr):
    order = get_order(order_id)

    if "error" in order:
        raise ValueError("Order not found.")

    if str(order["status"]).lower() != "completed":
        raise ValueError("Only completed orders can be refunded.")

    expected = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"]
    )["refund_inr"]

    if expected <= 0:
        raise ValueError("No refund is allowed for delivery within 45 minutes.")

    if round(float(refund_inr), 2) != expected:
        raise ValueError("Refund amount does not match policy.")

    return expected


def issue_refund(order_id, refund_inr):
    amount = _validate_refund(order_id, refund_inr)
    return _ledger("issue_refund", order_id, amount)


def escalate_to_human(order_id, refund_inr):
    amount = _validate_refund(order_id, refund_inr)

    if amount <= 1000:
        raise ValueError("Human escalation is only for amounts above INR 1000.")

    return _ledger("escalate_to_human", order_id, amount)
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Get order details using the order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the company policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund using order amount and delivery time in minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "integer"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an eligible refund for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a refund above INR 1000 to a human agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


def _save_checkpoint(run_id, scenario_id, messages, tokens,
                     steps, executed, llm_calls, step_number):
    STATE.mkdir(exist_ok=True)

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens,
        "step_number": step_number,
        "tools_executed": executed,
        "steps": steps,
        "llm_calls": llm_calls
    }

    path = STATE / f"{run_id}.json"
    path.write_text(
        json.dumps(checkpoint, default=str, indent=2),
        encoding="utf-8"
    )


def _log_run(result, run_id, scenario_id):
    OUT.mkdir(exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "scenario_id": scenario_id,
        **result
    }

    with (OUT / "agent_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _run_loop(run_id, scenario_id, messages, tokens, steps,
              executed, llm_calls, max_iterations,
              token_limit, time_limit_seconds, crash_after=None):

    STATE.mkdir(exist_ok=True)
    start = time.monotonic()
    initial_tool_count = len(executed)

    for iteration in range(max_iterations):

        if time.monotonic() - start >= time_limit_seconds:
            result = {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: hand off to a human agent.",
                "steps": steps,
                "tools_executed": executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }
            _log_run(result, run_id, scenario_id)
            return result

        response = mock_llm.chat(messages)
        llm_calls += 1
        tokens += response.usage.total_tokens

        if tokens > token_limit:
            result = {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: hand off to a human agent.",
                "steps": steps,
                "tools_executed": executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }
            _log_run(result, run_id, scenario_id)
            return result

        msg = response.choices[0].message

        if not msg.tool_calls:
            content = msg.content or ""
            status = content.split(":", 1)[0]

            result = {
                "status": status,
                "final_answer": content,
                "steps": steps,
                "tools_executed": executed,
                "llm_calls": llm_calls,
                "tokens_used": tokens
            }
            _log_run(result, run_id, scenario_id)
            return result

        tc = msg.tool_calls[0]
        args = tc.arguments

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tc.id,
                "name": tc.name,
                "arguments": args
            }]
        })

        observation = TOOL_MAP[tc.name](**args)
        executed.append(tc.name)

        steps.append({
            "tool": tc.name,
            "arguments": args,
            "observation": observation
        })

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(observation)
        })

        _save_checkpoint(
            run_id, scenario_id, messages, tokens,
            steps, executed, llm_calls, len(executed)
        )

        if (
            crash_after is not None
            and len(executed) - initial_tool_count == crash_after
        ):
            raise RuntimeError("Simulated crash after checkpoint")

    result = {
        "status": "FALLBACK_MAX_ITERATIONS",
"final_answer": "FALLBACK_MAX_ITERATIONS: maximum iterations reached; hand off to a human agent.",
        "steps": steps,
        "tools_executed": executed,
        "llm_calls": llm_calls,
        "tokens_used": tokens
    }

    _log_run(result, run_id, scenario_id)
    return result


def run_agent(user_message, scenario_id="demo", max_iterations=6,
              token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after=None):


    if scenario_id == "S7":
        user_message = "[LOOP]"

    run_id = run_id or str(uuid.uuid4())

    messages = [
        {
            "role": "system",
            "content": "AGENT: Use tools to verify orders and follow refund policy."
        },
        {"role": "user", "content": user_message}
    ]

    return _run_loop(
        run_id, scenario_id, messages,
        tokens=0,
        steps=[],
        executed=[],
        llm_calls=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after=crash_after
    )


def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):

    checkpoint_path = STATE / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for run_id: {run_id}"
        )

    checkpoint = json.loads(
        checkpoint_path.read_text(encoding="utf-8")
    )

    return _run_loop(
        run_id=run_id,
        scenario_id=checkpoint["scenario_id"],
        messages=checkpoint["messages"],
        tokens=checkpoint["tokens_used"],
        steps=checkpoint["steps"],
        executed=checkpoint["tools_executed"],
        llm_calls=checkpoint["llm_calls"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )