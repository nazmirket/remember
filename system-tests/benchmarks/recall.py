"""Measure warm direct recall against an explicitly selected test bank.

This sends read-only HTTP requests. It does not create a bank, seed data or measure
LLM-backed reflection; use synthetic data and record the server's configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="URL of the isolated test server")
    parser.add_argument("--bank", required=True, help="Existing synthetic bank ID")
    parser.add_argument("--query", required=True, help="Recall query to repeat")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    endpoint = f"{args.base_url.rstrip('/')}/v1/default/banks/{quote(args.bank, safe='')}/memories/recall"
    payload = json.dumps({"query": args.query}).encode()
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("REMEMBER_API_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    samples: list[float] = []
    result_counts: list[int] = []
    for _ in range(args.samples):
        request = Request(endpoint, data=payload, headers=headers)
        started = time.perf_counter()
        with urlopen(request, timeout=60) as response:
            body = json.load(response)
        samples.append((time.perf_counter() - started) * 1000)
        result_counts.append(len(body["results"]))

    ordered = sorted(samples)
    print(
        json.dumps(
            {
                "samples": len(samples),
                "unit": "ms",
                "p50": statistics.median(samples),
                "p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
                "min": min(samples),
                "max": max(samples),
                "result_counts": result_counts,
                "raw_ms": samples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
