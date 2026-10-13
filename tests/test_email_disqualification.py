import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
    from gtm_agent.gtm_agent import SYSTEM_PROMPT, get_prospect, send_prospect_email


class SendProspectEmailDisqualificationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_mrossi"}})
        self.from_rep = {"name": "Rep", "email": "rep@example.com"}
        self.prospect = {
            "prospect_id": "LEAD-12345",
            "name": "Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }

    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Demo",
            body="Let's talk.",
            runtime=self.runtime,
            from_rep=self.from_rep,
        )

    def test_blocks_authoritatively_disqualified_prospect(self):
        with (
            patch.object(
                data_service, "get_prospect_record", return_value={"disqualified": True}
            ),
            patch("gtm_agent.gtm_agent.uuid.uuid4") as make_uuid,
        ):
            result = self.send(self.prospect)

        self.assertEqual(
            result,
            {
                "status": "blocked",
                "error": "Prospect is disqualified; email not sent.",
                "prospect_id": "LEAD-12345",
            },
        )
        make_uuid.assert_not_called()

    def test_uses_authoritative_disqualification_over_supplied_data(self):
        with patch.object(
            data_service, "get_prospect_record", return_value={"disqualified": True}
        ):
            result = self.send({**self.prospect, "disqualified": False})

        self.assertEqual(result["status"], "blocked")

    def test_sends_when_authoritative_record_is_not_disqualified(self):
        with patch.object(
            data_service, "get_prospect_record", return_value={"disqualified": False}
        ):
            result = self.send(self.prospect)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], "prospect@example.com")

    def test_preserves_send_behavior_without_id_or_record(self):
        with patch.object(data_service, "get_prospect_record") as get_record:
            without_id = self.send({"email": "prospect@example.com"})
        get_record.assert_not_called()
        self.assertEqual(without_id["status"], "sent")

        with patch.object(data_service, "get_prospect_record", return_value=None):
            without_record = self.send(self.prospect)
        self.assertEqual(without_record["status"], "sent")

    def test_lead_50001_is_blocked_and_prompt_explains_disqualification(self):
        prospect = get_prospect.func("LEAD-50001")["prospect"]

        result = self.send(prospect)

        self.assertEqual(result["status"], "blocked")
        self.assertIn(
            "do not send, and tell the rep the prospect is disqualified",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
