#!/usr/bin/env python3
"""Test that a launcher works as an MCP server would be launched by the host.

Drives the real MCP handshake over stdio against whatever command it is given
and reports whether initialization and tool discovery succeed. Used both for the
bundle directory and the packed .mcpb, so the thing that ships is the thing that
was tested.

    python3 tests/test_mcpb_bundle.py <manifest.json> [--extract-to DIR]

Exits non-zero if the server does not complete initialization or does not
expose the expected tool surface.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = 50

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


def resolve(value: str, bundle_dir: Path, config: dict[str, str]) -> str:
    """Apply the same substitutions the MCP host applies."""
    out = value
    out = out.replace("${__dirname}", str(bundle_dir))
    out = out.replace("${HOME}", str(Path.home()))
    for key, replacement in config.items():
        out = out.replace("${user_config." + key + "}", replacement)
    return out


def run_handshake(command: str, args: list[str], env: dict[str, str],
                  cwd: Path) -> tuple[bool, dict]:
    """Speak the MCP handshake to a stdio server. Returns (ok, detail)."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "mcpb-verify", "version": "1.0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        # A real call, so this proves the server works and not merely that it
        # can describe itself.
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "list_projects", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "list_agents", "arguments": {}}},
    ]
    payload = "".join(json.dumps(r) + "\n" for r in requests)

    proc = subprocess.run(
        [command, *args], input=payload, capture_output=True, text=True,
        env={**os.environ, **env}, cwd=str(cwd), timeout=180,
    )

    detail: dict = {"stderr": proc.stderr[-2000:], "returncode": proc.returncode}

    lines = [line for line in proc.stdout.split("\n") if line.strip()]
    detail["response_lines"] = len(lines)

    responses = []
    for line in lines:
        try:
            responses.append(json.loads(line))
        except json.JSONDecodeError:
            # Anything non-JSON on stdout is a protocol corruption.
            detail["garbage_on_stdout"] = line[:200]

    init = next((r for r in responses if r.get("id") == 1), None)
    tools = next((r for r in responses if r.get("id") == 2), None)
    projects_call = next((r for r in responses if r.get("id") == 3), None)
    agents_call = next((r for r in responses if r.get("id") == 4), None)
    detail["init"] = init
    detail["projects_call"] = projects_call
    detail["agents_call"] = agents_call
    detail["tool_count"] = (
        len(tools["result"]["tools"]) if tools and "result" in tools else 0
    )
    if tools and "result" in tools:
        detail["tool_names"] = {t["name"] for t in tools["result"]["tools"]}

    def call_ok(response) -> bool:
        if not response or "result" not in response:
            return False
        result = response["result"]
        return (not result.get("isError")
                and bool(result.get("content"))
                and result["content"][0].get("type") == "text")

    detail["calls_ok"] = call_ok(projects_call) and call_ok(agents_call)

    ok = (
        init is not None
        and "result" in init
        and tools is not None
        and "result" in tools
        and detail["tool_count"] >= EXPECTED_TOOLS
        and detail["calls_ok"]
        and "garbage_on_stdout" not in detail
    )
    return ok, detail


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    manifest_path = Path(sys.argv[1]).resolve()
    manifest = json.loads(manifest_path.read_text())
    bundle_dir = manifest_path.parent

    config = {
        "project_dir": os.environ.get("FA_TEST_PROJECT_DIR", str(REPO_ROOT)),
        "python_path": os.environ.get("FA_TEST_PYTHON", sys.executable),
    }

    print(f"bundle : {bundle_dir}")
    print(f"config : project_dir={config['project_dir']}")
    print(f"         python_path={config['python_path']}")
    print()

    server = manifest["server"]
    mcp_config = server["mcp_config"]

    command = resolve(mcp_config["command"], bundle_dir, config)
    args = [resolve(a, bundle_dir, config) for a in mcp_config.get("args", [])]
    env = {k: resolve(v, bundle_dir, config)
           for k, v in (mcp_config.get("env") or {}).items()}

    print(f"launch : {command} {' '.join(args)}")
    for key, value in env.items():
        print(f"         {key}={value}")
    print()

    print("MCP handshake over stdio")
    ok, detail = run_handshake(command, args, env, bundle_dir)
    check("server responded to initialize", detail.get("init") is not None,
          str(detail.get("stderr", ""))[:300])

    init = detail.get("init") or {}
    result = init.get("result") or {}
    check("initialize returned a result", bool(result), json.dumps(init)[:300])
    check("server identifies itself as filmautomator",
          (result.get("serverInfo") or {}).get("name") == "filmautomator",
          str(result.get("serverInfo")))
    check("protocol version negotiated",
          result.get("protocolVersion") in {"2025-06-18", "2025-03-26", "2024-11-05"},
          str(result.get("protocolVersion")))
    check("tool capability advertised", "tools" in (result.get("capabilities") or {}))

    check(f"tool discovery returned at least {EXPECTED_TOOLS} tools",
          detail.get("tool_count", 0) >= EXPECTED_TOOLS,
          f"got {detail.get('tool_count')}")

    names = detail.get("tool_names") or set()
    for required in ("create_project", "start_production", "get_production_status",
                     "get_preview", "get_final_movie"):
        check(f"tool present: {required}", required in names)

    check("stdout carried only protocol messages",
          "garbage_on_stdout" not in detail,
          str(detail.get("garbage_on_stdout"))[:200])

    # Real tool calls, proving the packaged server does work rather than only
    # describing itself.
    for label, response in (("list_projects", detail.get("projects_call")),
                            ("list_agents", detail.get("agents_call"))):
        check(f"tool call succeeds: {label}",
              bool(response) and "result" in response
              and not response["result"].get("isError"),
              json.dumps(response)[:250] if response else "no response")

    # The bundle must not put anything on the network. stdio only.
    launch_args = " ".join(args)
    check("manifest does not enable a network transport",
          "--http-port" not in launch_args and "--http-only" not in launch_args,
          launch_args)
    check("manifest declares no port environment variables",
          not any("PORT" in k.upper() for k in env),
          str(list(env)))
    check("manifest contains no URL that would expose the server",
          "0.0.0.0" not in json.dumps(manifest),
          "0.0.0.0 found in manifest")

    print()
    print(f"  protocol : {result.get('protocolVersion')}")
    print(f"  server   : {(result.get('serverInfo') or {}).get('name')} "
          f"{(result.get('serverInfo') or {}).get('version')}")
    print(f"  tools    : {detail.get('tool_count')}")
    if detail.get("stderr", "").strip():
        print(f"  stderr   : {detail['stderr'].strip().splitlines()[-1][:160]}")

    print()
    print(f"{PASSED} passed, {FAILED} failed")
    return 0 if (ok and FAILED == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
