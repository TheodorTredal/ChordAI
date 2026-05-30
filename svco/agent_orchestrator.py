"""Anthropic + MCP orchestration for dual-channel chord/voice responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DUAL_CHANNEL_SYSTEM_PROMPT = (
    "You are a music assistant. Evaluate the user payload and use MCP tools for chord generation when needed. "
    "Your final response must be strict JSON with exactly two keys: "
    "speech_text (string, spoken English prose only) and chord_data (array of chord strings)."
)


@dataclass
class AgentOrchestrator:
    anthropic_api_key: str
    mcp_server_command: str
    mcp_server_args: Optional[List[str]] = None
    model: str = "claude-sonnet-4-5"

    def __post_init__(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic package is required.") from exc

        self._anthropic_client = Anthropic(api_key=self.anthropic_api_key)

    async def generate_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tools = await self._load_mcp_tools()

        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=DUAL_CHANNEL_SYSTEM_PROMPT,
            tools=tools,
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )

        raw_text = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                raw_text += getattr(block, "text", "")

        return parse_dual_channel_response(raw_text)

    async def _load_mcp_tools(self) -> List[Dict[str, Any]]:
        """Load tool schemas from an MCP stdio server."""
        try:
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise RuntimeError("mcp SDK package is required.") from exc

        server_params = StdioServerParameters(
            command=self.mcp_server_command,
            args=self.mcp_server_args or [],
        )

        tools: List[Dict[str, Any]] = []
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed_tools = await session.list_tools()
                for tool in getattr(listed_tools, "tools", []):
                    tools.append(
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.inputSchema,
                        }
                    )

        return tools
    


def parse_dual_channel_response(raw_response: str) -> Dict[str, Any]:
    data = json.loads(raw_response)

    speech_text = data.get("speech_text")
    chord_data = data.get("chord_data")

    if not isinstance(speech_text, str):
        raise ValueError("`speech_text` must be a string.")
    if not isinstance(chord_data, list) or not all(isinstance(item, str) for item in chord_data):
        raise ValueError("`chord_data` must be an array of strings.")

    return {"speech_text": speech_text, "chord_data": chord_data}
