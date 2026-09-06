import json

import pytest

from sfsc.contracts import ContractError
from sfsc.model_worker import build_payload, parse_response


def test_research_requires_academic_search_and_implementation_uses_no_search():
    request = {"instructions": "A task", "role": "research"}
    payload = build_payload(request, "chosen-model")
    assert payload["model"] == "chosen-model"
    assert payload["tool_choice"] == "required"
    assert "arxiv.org" in payload["tools"][0]["filters"]["allowed_domains"]
    assert payload["store"] is False
    request["role"] = "implementation"
    assert "tools" not in build_payload(request, "chosen-model")


def test_response_parser_retains_model_and_search_evidence():
    response = {
        "status": "completed",
        "model": "chosen-model",
        "id": "test-response",
        "output": [
            {"type": "web_search_call", "action": {"sources": []}},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps({"approaches": []})}],
            },
        ],
    }
    parsed = parse_response(response, research_role=True)
    assert parsed["worker_provenance"]["model"] == "chosen-model"
    assert len(parsed["worker_provenance"]["searches"]) == 1


@pytest.mark.parametrize(
    "response", [{"status": "incomplete"}, {"status": "completed", "output": []}]
)
def test_incomplete_or_unresearched_output_rejected(response):
    with pytest.raises(ContractError):
        parse_response(response, research_role=True)
