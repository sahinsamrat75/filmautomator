"""The Model Gateway must route by capability and degrade honestly."""

from __future__ import annotations

import json

from filmautomator.gateway import Capability, ModelGateway
from filmautomator.gateway.capabilities import Capability as Cap
from filmautomator.gateway.drivers.mock import MockDriver, synthesize_from_schema
from filmautomator.gateway.drivers.openai_compat import OpenAICompatDriver, _extract_json


# -- JSON extraction -------------------------------------------------------
# Local models wrap JSON in prose and code fences constantly. Getting this
# wrong turns every structured request into a failed task.


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = 'Here you go:\n```json\n{"a": 1, "b": [2]}\n```\nHope that helps!'
    assert _extract_json(text) == {"a": 1, "b": [2]}


def test_extract_json_with_prose_around_it():
    text = 'Sure! The plan is {"shots": [{"duration_s": 4}]} and that is all.'
    assert _extract_json(text) == {"shots": [{"duration_s": 4}]}


def test_extract_json_ignores_braces_inside_strings():
    text = '{"note": "use {curly} braces", "n": 3}'
    assert _extract_json(text) == {"note": "use {curly} braces", "n": 3}


def test_extract_json_handles_escaped_quotes():
    text = r'{"quote": "he said \"stop\""}'
    assert _extract_json(text) == {"quote": 'he said "stop"'}


def test_extract_json_returns_none_for_prose():
    assert _extract_json("I could not do that.") is None
    assert _extract_json("") is None


def test_extract_json_takes_first_valid_object():
    text = 'prefix {"first": 1} then {"second": 2}'
    assert _extract_json(text) == {"first": 1}


# -- schema synthesis ------------------------------------------------------


def test_synthesize_honours_required_and_enum():
    schema = {
        "type": "object",
        "properties": {
            "shot_size": {"type": "string", "enum": ["medium", "close_up"]},
            "duration_s": {"type": "number", "minimum": 1.0, "maximum": 30.0},
            "optional": {"type": "string"},
        },
        "required": ["shot_size", "duration_s"],
    }
    result = synthesize_from_schema(schema)
    assert result["shot_size"] == "medium"
    assert 1.0 <= result["duration_s"] <= 30.0
    assert "optional" not in result


def test_synthesize_clamps_to_bounds():
    schema = {"type": "number", "minimum": 5.0, "maximum": 10.0}
    assert 5.0 <= synthesize_from_schema(schema) <= 10.0


def test_synthesize_nested_objects_and_arrays():
    schema = {
        "type": "object",
        "properties": {
            "shots": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            }
        },
        "required": ["shots"],
    }
    result = synthesize_from_schema(schema)
    assert isinstance(result["shots"], list)
    assert "id" in result["shots"][0]


# -- routing ---------------------------------------------------------------


def test_unreachable_endpoint_falls_back_to_synthetic(gateway: ModelGateway):
    response = gateway.invoke(Capability.PLANNING, "plan something")
    assert response.synthetic is True
    assert response.driver == "mock"


def test_synthetic_response_carries_schema_shaped_data(gateway: ModelGateway):
    schema = {
        "type": "object",
        "properties": {"passed": {"type": "boolean"}},
        "required": ["passed"],
    }
    response = gateway.invoke(Capability.VISION, "judge this", json_schema=schema)
    assert response.synthetic is True
    assert response.data == {"passed": False}


def test_mock_never_advertised_as_a_real_provider(gateway: ModelGateway):
    report = gateway.report()
    for providers in report.capability_map.values():
        assert "mock" not in providers


def test_gateway_reports_unimplemented_capabilities(gateway: ModelGateway):
    report = gateway.report()
    # Video generation has no free local driver yet and must say so.
    assert "video_generation" in report.unimplemented


def test_supports_is_false_when_nothing_real_is_available(gateway: ModelGateway):
    assert gateway.supports(Capability.VISION) is False


def test_mock_driver_claims_every_capability():
    driver = MockDriver()
    assert driver.capabilities() == frozenset(Cap)


# -- OpenAI-compatible driver ---------------------------------------------


def test_driver_reports_unavailable_when_server_is_down():
    driver = OpenAICompatDriver(base_url="http://127.0.0.1:9/v1", timeout_s=0.5)
    assert driver.available() is False
    assert driver.capabilities() == frozenset()
    assert "unreachable" in driver.describe() or "no models" in driver.describe()


def test_vision_name_heuristic():
    assert OpenAICompatDriver._looks_like_vision("qwen2-vl-7b-instruct")
    assert OpenAICompatDriver._looks_like_vision("llava-v1.6-mistral-7b")
    assert OpenAICompatDriver._looks_like_vision("gemma-3-4b-it")
    assert not OpenAICompatDriver._looks_like_vision("qwen2.5-7b-instruct")
    assert not OpenAICompatDriver._looks_like_vision("llama-3.1-8b")


def test_model_override_wins_over_discovery():
    driver = OpenAICompatDriver(
        base_url="http://127.0.0.1:9/v1",
        model_overrides={"vision": "my-custom-vlm"},
    )
    assert driver._pick_model(Capability.VISION) == "my-custom-vlm"


def test_no_usable_model_raises_driver_unavailable():
    from filmautomator.gateway.base import DriverUnavailable, ModelRequest

    driver = OpenAICompatDriver(base_url="http://127.0.0.1:9/v1", timeout_s=0.5)
    try:
        driver.invoke(ModelRequest(capability=Capability.REASONING, prompt="hi"))
    except DriverUnavailable:
        return
    raise AssertionError("expected DriverUnavailable when no model can be selected")
