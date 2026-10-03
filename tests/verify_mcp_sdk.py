"""Independent interoperability check against the official MCP SDK.

The rest of the MCP suite drives this server with hand-written JSON-RPC, which
proves it matches *our* reading of the spec. This script proves something
stronger: that a real, independently implemented MCP client can discover and
call the server without any accommodation.

It runs the official ``mcp`` package as a client over stdio. That package is
deliberately NOT a dependency of Filmautomator — this is a test-time check, run
from a throwaway virtualenv:

    python3 -m venv /tmp/mcp_verify
    /tmp/mcp_verify/bin/pip install mcp
    /tmp/mcp_verify/bin/python tests/verify_mcp_sdk.py

Exit code 0 means a third-party client connected and operated the server.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}  {detail}")


async def run(workspace: Path) -> int:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        print("The official 'mcp' package is not installed in this interpreter.")
        print("This check is meant to be run from a virtualenv that has it:")
        print("  python3 -m venv /tmp/mcp_verify")
        print("  /tmp/mcp_verify/bin/pip install mcp")
        print("  /tmp/mcp_verify/bin/python tests/verify_mcp_sdk.py")
        return 2

    env = dict(os.environ)
    env["FA_WORKSPACE"] = str(workspace)
    env["PYTHONPATH"] = str(REPO_ROOT)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "filmautomator", "mcp"],
        env=env,
        cwd=str(REPO_ROOT),
    )

    print("connecting the official MCP SDK client to the Filmautomator server")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            check("initialize handshake completes", init is not None)
            server_info = getattr(init, "serverInfo", None)
            name = getattr(server_info, "name", "") if server_info else ""
            check("server identifies itself", name == "filmautomator", f"got {name!r}")
            print(f"        protocol {init.protocolVersion}, server {name} "
                  f"{getattr(server_info, 'version', '?')}")

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            check("SDK discovers the tool set", len(names) >= 50, f"{len(names)} tools")
            check("core tools are present",
                  {"create_project", "start_production", "get_preview",
                   "get_final_movie"} <= names)
            print(f"        discovered {len(names)} tools")

            # A real tool call, through the real client.
            created = await session.call_tool(
                "create_project",
                {"name": "SDK_VERIFY", "objective": "interop check",
                 "duration_s": 5},
            )
            check("tool call succeeds", not created.isError)
            payload = json.loads(created.content[0].text)
            project_id = payload.get("project_id", "")
            check("project is created", project_id.startswith("proj_"),
                  str(payload)[:200])

            listed = await session.call_tool("list_projects", {})
            listing = json.loads(listed.content[0].text)
            check("the created project is listed",
                  any(p["project_id"] == project_id
                      for p in listing.get("projects", [])))

            status = await session.call_tool(
                "get_production_status", {}
            )
            state = json.loads(status.content[0].text)
            check("status reports idle", state.get("state") == "IDLE",
                  str(state)[:200])

            # An error path, to confirm failures are reported rather than
            # dropping the connection.
            bad = await session.call_tool("get_project",
                                          {"project_id": "proj_missing"})
            check("a failed call is reported as a tool error", bad.isError)

            # The connection must still work after that.
            again = await session.call_tool("list_projects", {})
            check("connection survives a failed call", not again.isError)

    return 0


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="fa_sdk_verify_"))
    print(f"workspace: {workspace}")
    print()
    try:
        asyncio.run(run(workspace))
    except Exception as exc:  # noqa: BLE001
        print(f"\nfailed: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        globals()["FAILED"] = FAILED + 1

    print()
    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
