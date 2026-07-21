"""
Local model solver: communicates with llama.cpp server (Qwythos-9B).

The model is loaded by llama.cpp server running in the same container.
Communicates via stdlib urllib to the REST API (OpenAI-compatible).

Key features:
- Native function calling (Qwythos's <tool_call> format)
- Multi-round tool execution (model calls tool -> we execute -> return result)
- Response cleaning (strip preambles, code fence artifacts)
- Deadline-based escalation (if model takes too long, signal Fireworks fallback)
"""

import json
import logging
import re
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from agent.config import LLAMA_SERVER_URL
from agent.answer_cleaner import clean_response
from agent.solvers.tools import execute_tool, get_tool_schemas

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3  # Qwen3.5-4B is smaller, fewer rounds needed
LOCAL_DEADLINE_SEC = 180  # 3-minute deadline for CPU inference on complex tasks


class LocalSolver:
    """Solver that uses the local llama.cpp model (Qwythos-9B)."""

    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url

    def _completion(
        self,
        messages: List[Dict],
        tools: Optional[List[dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: int = 15,
        reasoning: bool = True,
    ) -> Dict:
        """Call the llama.cpp server's chat completions endpoint via urllib."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|im_end|>"],
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            logger.error(f"llama.cpp HTTP {e.code}: {e.read().decode()[:200]}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"llama.cpp connection error: {e.reason}")
            raise

    def _parse_tool_calls(self, content: str) -> List[Dict]:
        """
        Parse tool calls from model output.
        Supports both native <tool_call> format and simplified JSON format.
        """
        tool_calls = []
        # Try native <tool_call> format
        pattern = r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>'
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            params_text = match.group(2).strip()

            args = {}
            param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*</parameter>'
            for p_match in re.finditer(param_pattern, params_text, re.DOTALL):
                param_name = p_match.group(1)
                param_value = p_match.group(2).strip()
                args[param_name] = param_value

            tool_calls.append({"name": func_name, "arguments": args})

        # Also try JSON function call format
        if not tool_calls:
            json_pattern = r'\{\s*"function"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}'
            for match in re.finditer(json_pattern, content, re.DOTALL):
                try:
                    obj = json.loads(match.group(0))
                    tool_calls.append({"name": obj["function"], "arguments": obj.get("arguments", {})})
                except (json.JSONDecodeError, KeyError):
                    pass

        return tool_calls

    def solve(
        self,
        system_prompt: str,
        user_prompt: str,
        tools_enabled: bool = True,
        reasoning: bool = True,
    ) -> Optional[str]:
        """
        Solve a task using the local model.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User task prompt
            tools_enabled: Whether to enable tool calls
            reasoning: Whether to allow chain-of-thought reasoning.
                       Disable for simple tasks (sentiment, NER, factual)
                       to save tokens and get cleaner output.

        Returns the answer string, or None if it hit the deadline
        (triggers Fireworks fallback in the orchestrator).
        """
        start = time.monotonic()

        messages = [
            {"role": "system", "content": system_prompt or "You are a precise AI assistant."},
            {"role": "user", "content": user_prompt},
        ]
        tools = get_tool_schemas() if tools_enabled else None

        for round_num in range(MAX_TOOL_ROUNDS + 1):
            # Check deadline
            if time.monotonic() - start > LOCAL_DEADLINE_SEC:
                logger.warning(f"Local solver deadline exceeded ({LOCAL_DEADLINE_SEC}s)")
                return None

            remaining = LOCAL_DEADLINE_SEC - (time.monotonic() - start)
            timeout = min(90, max(15, int(remaining)))

            try:
                response = self._completion(messages, tools=tools, timeout=timeout, reasoning=reasoning)
            except Exception as e:
                logger.warning(f"Local model call failed: {e}")
                return None

            choice = response["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            # Qwen3.5-4B sometimes puts the answer in reasoning_content
            # instead of content for longer prompts. Fall back to it.
            if not content.strip():
                content = message.get("reasoning_content", "") or ""
            finish_reason = choice.get("finish_reason", "")

            # Check for OpenAI-format tool calls first (llama.cpp b9937+)
            raw_tool_calls = message.get("tool_calls")
            if raw_tool_calls:
                tool_calls = []
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_raw
                    tool_calls.append({"name": name, "arguments": args})

                if tool_calls:
                    messages.append({"role": "assistant", "content": content or json.dumps(raw_tool_calls)})
                    for tc in tool_calls:
                        name = tc["name"]
                        args = tc["arguments"]
                        result = execute_tool(name, args)
                        messages.append({
                            "role": "user",
                            "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>",
                        })
                    continue

            # No tool calls -> final answer
            if content.strip():
                return clean_response(content)

            # If finish_reason is tool_calls but we didn't parse any, try content-based parsing
            if finish_reason == "tool_calls":
                tool_calls = self._parse_tool_calls(content)
                if tool_calls:
                    messages.append({"role": "assistant", "content": content})
                    for tc in tool_calls:
                        name = tc["name"]
                        args = tc["arguments"]
                        result = execute_tool(name, args)
                        messages.append({
                            "role": "user",
                            "content": f"<tool_response>\n{json.dumps(result)}\n</tool_response>",
                        })
                    continue

            return None

        # Max rounds reached - return whatever we have
        return None
