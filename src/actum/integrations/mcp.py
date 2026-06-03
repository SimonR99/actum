"""Optional MCP client bridge for configured external tool servers."""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass
class MCPServerSpec:
    name: str
    enabled: bool
    transport: str
    command: str = ""
    args: list[str] | None = None
    url: str = ""
    env: dict[str, str] | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "transport": self.transport,
            "command": self.command,
            "args": list(self.args or []),
            "url": self.url,
            "env_keys": sorted((self.env or {}).keys()),
        }


def sdk_available() -> bool:
    return importlib.util.find_spec("mcp") is not None


def configured_servers(config: dict[str, Any]) -> list[MCPServerSpec]:
    mcp_cfg = config.get("mcp", {}) if isinstance(config.get("mcp"), dict) else {}
    if not bool(mcp_cfg.get("enabled", False)):
        return []

    raw_servers = mcp_cfg.get("servers", {})
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_servers, dict):
        for name, value in raw_servers.items():
            if isinstance(value, dict):
                items.append((str(name), value))
    elif isinstance(raw_servers, list):
        for value in raw_servers:
            if isinstance(value, dict) and value.get("name"):
                items.append((str(value["name"]), value))

    specs: list[MCPServerSpec] = []
    for name, server in items:
        transport = str(server.get("transport", "stdio")).lower().strip().replace("-", "_")
        env = server.get("env", {}) if isinstance(server.get("env"), dict) else {}
        specs.append(
            MCPServerSpec(
                name=name,
                enabled=bool(server.get("enabled", True)),
                transport=transport,
                command=str(server.get("command", "")),
                args=[str(item) for item in server.get("args", [])] if isinstance(server.get("args", []), list) else [],
                url=str(server.get("url", "")),
                env={str(key): str(value) for key, value in env.items()},
            )
        )
    return specs


def describe_servers(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [server.public_dict() for server in configured_servers(config)]


def list_tools(config: dict[str, Any], server_name: str) -> dict[str, Any]:
    server = _select_server(config, server_name)
    return _run(_list_tools_async(server))


def call_tool(config: dict[str, Any], server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    server = _select_server(config, server_name)
    return _run(_call_tool_async(server, tool_name, arguments))


async def _list_tools_async(server: MCPServerSpec) -> dict[str, Any]:
    async def operation(session: Any) -> dict[str, Any]:
        response = await session.list_tools()
        return {"server": server.name, "tools": [_tool_to_dict(tool) for tool in response.tools]}

    return await _with_session(server, operation)


async def _call_tool_async(server: MCPServerSpec, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def operation(session: Any) -> dict[str, Any]:
        result = await session.call_tool(tool_name, arguments)
        return {"server": server.name, "tool": tool_name, "result": _result_to_dict(result)}

    return await _with_session(server, operation)


async def _with_session(
    server: MCPServerSpec,
    operation: Callable[[Any], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    if not server.enabled:
        raise RuntimeError(f"MCP server {server.name!r} is disabled.")
    if not sdk_available():
        raise RuntimeError("MCP SDK is not installed. Install with `pip install -e .[mcp]`.")

    if server.transport == "stdio":
        if not server.command:
            raise ValueError(f"MCP server {server.name!r} is missing command.")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=server.command, args=server.args or [], env=server.env or None)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    if server.transport in {"http", "streamable_http"}:
        if not server.url:
            raise ValueError(f"MCP server {server.name!r} is missing url.")
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(server.url) as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await operation(session)

    raise ValueError(f"Unsupported MCP transport for {server.name!r}: {server.transport!r}")


def _select_server(config: dict[str, Any], server_name: str) -> MCPServerSpec:
    requested = str(server_name).strip()
    for server in configured_servers(config):
        if server.name == requested:
            return server
    raise KeyError(f"Configured MCP server not found: {requested!r}")


def _run(coro: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    close = getattr(coro, "close", None)
    if callable(close):
        close()
    raise RuntimeError("MCP bridge cannot run synchronously inside an active event loop.")


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "") or ""),
        "input_schema": getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {},
    }


def _result_to_dict(result: Any) -> dict[str, Any]:
    content = []
    for block in getattr(result, "content", []) or []:
        content.append(
            {
                "type": getattr(block, "type", ""),
                "text": getattr(block, "text", ""),
                "mime_type": getattr(block, "mimeType", "") or getattr(block, "mime_type", ""),
            }
        )
    return {
        "content": content,
        "structured_content": getattr(result, "structuredContent", None)
        or getattr(result, "structured_content", None),
        "is_error": bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
    }
