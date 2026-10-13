import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


class SensitiveProspectDataTests(unittest.TestCase):
    def test_get_prospect_returns_only_allowlisted_fields(self):
        record = {
            "name": "Avery Example",
            "email": "avery@example.com",
            "disqualified": False,
            "billing_qualification": {"tax_id": "123", "card_on_file": "456"},
            "annual_revenue": 1000000,
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = get_prospect.func("LEAD-1")

        self.assertEqual(
            result,
            {
                "prospect": {
                    "prospect_id": "LEAD-1",
                    "name": "Avery Example",
                    "email": "avery@example.com",
                    "disqualified": False,
                },
                "found": True,
            },
        )
        self.assertNotIn("tax_id", json.dumps(result))
        self.assertNotIn("card_on_file", json.dumps(result))

    def test_build_prospect_profile_excludes_sensitive_fields(self):
        record = {
            "name": "Avery Example",
            "email": "avery@example.com",
            "billing_qualification": {"tax_id": "123", "card_on_file": "456"},
            "annual_revenue": 1000000,
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(data_service, "save_profile_to_db"),
        ):
            result = build_prospect_profile.func("LEAD-1")

        self.assertNotIn("billing_qualification", result["prospect_profile"])
        self.assertNotIn("tax_id", json.dumps(result))
        self.assertNotIn("card_on_file", json.dumps(result))

    def test_cached_profiles_are_sanitized_before_return(self):
        profile = {
            "prospect_id": "LEAD-1",
            "billing_qualification": {"tax_id": "123", "card_on_file": "456"},
        }
        with patch.dict(data_service._PROFILES, {"LEAD-1": profile}):
            result = build_prospect_profile.func("LEAD-1")["prospect_profile"]
            cached = data_service._PROFILES["LEAD-1"]

        self.assertNotIn("billing_qualification", result)
        self.assertNotIn("billing_qualification", cached)
        self.assertNotIn("tax_id", json.dumps(result))
        self.assertNotIn("card_on_file", json.dumps(result))

    def test_scoring_uses_server_data_and_never_accepts_a_profile_argument(self):
        prospect_id = "LEAD-1"
        record = {
            "billing_qualification": {"tax_id": "123", "card_on_file": "456"},
            "annual_revenue": 90000000,
            "segment": "Enterprise",
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1000000,
            "description": "A platform",
        }
        score = {"score": 85}
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(data_service, "fetch_account_details", return_value=[{"industry": "Software"}]),
            patch(
                "gtm_agent.gtm_agent._scoring_llm",
                new=Mock(invoke=Mock(return_value=Mock(model_dump=lambda: score))),
            ) as scoring_llm,
        ):
            result = score_prospect.func(prospect_id, offering)

        self.assertEqual(result, score)
        self.assertEqual(set(score_prospect.args), {"prospect_id", "offering"})
        user_content = scoring_llm.invoke.call_args.args[0][1]["content"]
        scoring_profile = json.loads(user_content.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            scoring_profile,
            {
                "annual_revenue": 90000000,
                "tech_stack": ["Snowflake"],
                "account_details": [{"industry": "Software"}],
                "segment": "Enterprise",
            },
        )
        for serialized in (user_content, json.dumps(result)):
            self.assertNotIn("tax_id", serialized)
            self.assertNotIn("card_on_file", serialized)


if __name__ == "__main__":
    unittest.main()
