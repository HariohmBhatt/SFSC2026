"""Optional OpenAI Responses adapter for the fleet's JSON stdin/stdout protocol."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from sfsc.contracts import ContractError, require

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "openreview.net",
    "proceedings.neurips.cc",
    "proceedings.mlr.press",
    "openaccess.thecvf.com",
    "nature.com",
    "science.org",
    "aanda.org",
    "academic.oup.com",
    "link.springer.com",
    "ieeexplore.ieee.org",
    "sciencedirect.com",
    "journals.aps.org",
    "iopscience.iop.org",
    "ui.adsabs.harvard.edu",
    "jmlr.org",
]


def build_payload(request: dict, model: str) -> dict:
    payload = {
        "model": model,
        "store": False,
        "instructions": request["instructions"]
        + (
            "\nReturn a single valid JSON object, without markdown fences or extra prose. "
            "The response_contract describes its shape; replace example values with your result."
        ),
        "input": json.dumps({k: v for k, v in request.items() if k != "instructions"}),
    }
    if request["role"] in {"research", "review"}:
        payload["tools"] = [
            {"type": "web_search", "filters": {"allowed_domains": ACADEMIC_DOMAINS}}
        ]
        payload["tool_choice"] = "required"
        payload["include"] = ["web_search_call.action.sources"]
    return payload


def parse_response(response: dict, research_role: bool) -> dict:
    require(response.get("status") == "completed", "Model response is incomplete or failed")
    output = response.get("output", [])
    if research_role:
        require(
            any(item.get("type") == "web_search_call" for item in output),
            "Research/review response must contain an academic web-search call",
        )
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    result = json.loads(text)
    require(isinstance(result, dict), "Model must return one JSON object")
    result["worker_provenance"] = {
        "provider": "openai",
        "model": response.get("model"),
        "response_id": response.get("id"),
        "usage": response.get("usage"),
        "searches": [item for item in output if item.get("type") == "web_search_call"],
    }
    return result


def main() -> None:
    try:
        request = json.load(sys.stdin)
        key = os.environ.get("OPENAI_API_KEY")
        role_key = f"SFSC_{request['role'].upper()}_MODEL"
        model = os.environ.get(role_key) or os.environ.get("SFSC_MODEL")
        require(bool(key), "Set OPENAI_API_KEY in the environment")
        require(bool(model), f"Set {role_key} or SFSC_MODEL to your chosen API model ID")
        payload = build_payload(request, model)
        http = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http, timeout=900) as response:
            result = parse_response(json.load(response), request["role"] in {"research", "review"})
        print(json.dumps(result, allow_nan=False))
    except urllib.error.HTTPError as error:
        sys.exit(f"OpenAI API returned HTTP {error.code}; check model access and account limits")
    except (ContractError, ValueError, KeyError, OSError) as error:
        sys.exit(f"sfsc-model-worker: {error}")


if __name__ == "__main__":
    main()
