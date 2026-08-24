import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


class StubFastAPI:
    def __init__(self, **_kwargs):
        pass

    def get(self, *_args, **_kwargs):
        return lambda function: function

    def post(self, *_args, **_kwargs):
        return lambda function: function


fastapi = types.ModuleType("fastapi")
fastapi.FastAPI = StubFastAPI
fastapi.Header = lambda default=None: default
fastapi.HTTPException = type("HTTPException", (Exception,), {})
fastapi.Request = type("Request", (), {})
responses = types.ModuleType("fastapi.responses")
responses.JSONResponse = type("JSONResponse", (), {})
httpx = types.ModuleType("httpx")
httpx.AsyncClient = type("AsyncClient", (), {})
sys.modules.setdefault("fastapi", fastapi)
sys.modules.setdefault("fastapi.responses", responses)
sys.modules.setdefault("httpx", httpx)


MODULE_PATH = Path(__file__).parents[1] / "app" / "main.py"
SPEC = importlib.util.spec_from_file_location("ha_llm_router_main", MODULE_PATH)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


def entity(entity_id, state, **attributes):
    return {
        "entity_id": entity_id,
        "state": state,
        "last_changed": "2026-08-13T12:00:00+00:00",
        "attributes": attributes,
    }


class RouterEntityTests(unittest.TestCase):
    def setUp(self):
        self.states = [
            entity(
                "climate.downstairs",
                "heat",
                friendly_name="Downstairs Thermostat",
                current_temperature=70,
                temperature=72,
                hvac_action="heating",
                hvac_modes=["off", "heat", "cool"],
                preset_mode="home",
                min_temp=45,
                max_temp=90,
            ),
            entity(
                "lawn_mower.backyard",
                "mowing",
                friendly_name="Backyard Mower",
                activity="mowing",
                battery_level=76,
                error_code=0,
            ),
            entity(
                "sensor.backyard_mower_battery",
                "76",
                friendly_name="Backyard Mower Battery",
                device_class="battery",
                unit_of_measurement="%",
            ),
            entity("light.kitchen", "off", friendly_name="Kitchen Light"),
        ]

    def select(self, text, mode):
        with patch.object(
            router, "fetch_ha_states", AsyncMock(return_value=self.states)
        ):
            return asyncio.run(router.select_relevant_entities(text, mode))

    def test_climate_status_selects_climate_and_keeps_operating_details(self):
        text = "Is the downstairs furnace currently heating?"
        self.assertEqual(router.classify_request(text), "status")

        selected = self.select(text, "status")

        self.assertEqual([item["entity_id"] for item in selected], ["climate.downstairs"])
        attrs = selected[0]["attributes"]
        self.assertEqual(attrs["hvac_action"], "heating")
        self.assertEqual(attrs["current_temperature"], 70)
        self.assertEqual(attrs["temperature"], 72)
        self.assertEqual(attrs["preset_mode"], "home")

    def test_ac_hint_matches_whole_word_only(self):
        self.assertEqual(router.extract_domain_hints("Set the AC to 72"), ["climate"])
        self.assertNotIn("climate", router.extract_domain_hints("Is the mower active?"))

    def test_mower_status_selects_lawn_mower(self):
        text = "What is the backyard mower doing?"
        self.assertEqual(router.classify_request(text), "status")

        selected = self.select(text, "status")

        self.assertEqual(selected[0]["entity_id"], "lawn_mower.backyard")
        self.assertEqual(selected[0]["attributes"]["activity"], "mowing")
        self.assertEqual(selected[0]["attributes"]["battery_level"], 76)

    def test_mower_battery_query_can_include_mower_and_battery_sensor(self):
        selected = self.select("What is the backyard mower battery?", "status")
        entity_ids = {item["entity_id"] for item in selected}

        self.assertIn("lawn_mower.backyard", entity_ids)
        self.assertIn("sensor.backyard_mower_battery", entity_ids)

    def test_mower_control_classification_and_tool_filtering(self):
        text = "Send the backyard mower home"
        self.assertEqual(router.classify_request(text), "control")
        tools = [
            {"function": {"name": "HassLawnMowerDock", "description": "Dock mower"}},
            {"function": {"name": "HassLightTurnOn", "description": "Turn on light"}},
        ]

        filtered = router.filter_tools_for_request(tools, text, "control")

        self.assertEqual(filtered, [tools[0]])

    def test_climate_control_filters_to_climate_tools(self):
        text = "Set the thermostat to 72"
        tools = [
            {"function": {"name": "HassClimateSetTemperature", "description": "Set climate temperature"}},
            {"function": {"name": "HassLightTurnOn", "description": "Turn on light"}},
        ]

        filtered = router.filter_tools_for_request(tools, text, "control")

        self.assertEqual(router.classify_request(text), "control")
        self.assertEqual(filtered, [tools[0]])


if __name__ == "__main__":
    unittest.main()
