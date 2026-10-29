#!/usr/bin/env python3
"""Minimal G0 smoke‑test driver for SilentFail‑Hardener.

- Instantiates :class:`src.silentfail.nebius.NebiusClient`.
- Calls :meth:`NebiusClient.probe_gate_g0` **only** when a real, non‑placeholder
  ``NEBIUS_API_KEY`` is present (the client reports ``is_live == True``).
- Writes a sanitized JSON evidence file into ``g0_results/`` with a UUID‑based
  filename.  The recorded fields are:
    * ``status`` – ``connected`` or ``local_simulation``
    * ``model`` – name of the Nemotron model used for the ping (if connected)
    * ``latency_ms`` – measured round‑trip time (ms)
    * ``measured_rps`` – derived requests‑per‑second estimate
    * ``token_usage`` – optional token count if the response contains it; omitted
      otherwise.
- No secrets, raw responses, or credentials are printed or written.
- Exits early with a clear message when the client is not live.
"""

import json
import os
import sys
import uuid
from pathlib import Path

# Import the client from the repository package
from src.silentfail.nebius import NebiusClient


def main() -> None:
    client = NebiusClient()
    # If the client could not initialise a live connection, skip G0.
    if not client.is_live:
        print(
            "[G0] No live Nebius configuration detected (NEBIUS_API_KEY missing or placeholder). "
            "Skipping connectivity test."
        )
        sys.exit(0)

    # Perform the single ping request.
    result = client.probe_gate_g0()

    # Build the sanitized evidence dictionary.
    evidence = {
        "status": result.get("status"),
        "model": result.get("model"),
        "latency_ms": result.get("latency_ms"),
        "measured_rps": result.get("measured_rps"),
    }
    # Some back‑ends may include token usage inside the raw response payload.
    # We deliberately avoid exposing the raw ``response`` field.
    if isinstance(result.get("response"), dict):
        usage = result["response"].get("usage")
        if usage:
            # Keep only safe numeric counters.
            evidence["token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
    # Ensure the output directory exists.
    out_dir = Path(__file__).parent / "g0_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename.
    filename = f"g0_{uuid.uuid4().hex}.json"
    out_path = out_dir / filename

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)

    print(f"[G0] Evidence written to: {out_path}")


if __name__ == "__main__":
    main()
