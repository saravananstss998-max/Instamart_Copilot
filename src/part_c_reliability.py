
from pathlib import Path
import time

# Configuration
MAX_RETRIES = 3
MAX_STEPS = 6
TOKEN_BUDGET = 6000

# Simulated tool call with retry handling
def call_tool(tool_function):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = tool_function()
            return result
        except Exception as error:
            print(f"Attempt {attempt} failed: {error}")

            if attempt == MAX_RETRIES:
                return {
                    "status": "failed",
                    "message": "Tool failed after maximum retries"
                }

            time.sleep(1)

# Token budget control
def check_budget(tokens_used, new_tokens):
    if tokens_used + new_tokens > TOKEN_BUDGET:
        return False
    return True

# Agent step and escalation control
def process_agent_steps(steps, tokens_used):
    if steps >= MAX_STEPS:
        return "ESCALATE_TO_HUMAN"

    if tokens_used >= TOKEN_BUDGET:
        return "BUDGET_LIMIT_REACHED"

    return "CONTINUE"

# Demonstration
print("===== PART C RELIABILITY TEST =====")

tokens_used = 1500
new_tokens = 300

print("Token budget:", TOKEN_BUDGET)
print("Current tokens:", tokens_used)
print("Can continue:", check_budget(tokens_used, new_tokens))

print("\nAgent step test:")
print("After 4 steps:", process_agent_steps(4, 1500))
print("After 6 steps:", process_agent_steps(6, 1500))

print("\nRetry test:")

attempt_count = 0

def sample_tool():
    global attempt_count
    attempt_count += 1

    if attempt_count < 3:
        raise Exception("Temporary service error")

    return "Tool call successful"

result = call_tool(sample_tool)
print("Final result:", result)

print("\nPart C reliability tests completed!")
