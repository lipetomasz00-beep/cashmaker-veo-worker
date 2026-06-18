import os
import sys
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# DYNAMIC SYS.MODULES MOCKS FOR SANDBOX RUNS
# ---------------------------------------------------------------------------
# Mock PIL/Pillow
mock_pil = MagicMock()
mock_image = MagicMock()
mock_draw = MagicMock()
mock_font = MagicMock()

sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_image
sys.modules['PIL.ImageDraw'] = mock_draw
sys.modules['PIL.ImageFont'] = mock_font

# Mock elevenlabs
mock_elevenlabs = MagicMock()
sys.modules['elevenlabs'] = mock_elevenlabs
sys.modules['elevenlabs.client'] = mock_elevenlabs

# Mock openai
mock_openai = MagicMock()
sys.modules['openai'] = mock_openai

# Mock google-genai (just in case)
# Since google-genai is already installed, we don't need to mock it, 
# but we mock any complex parts if needed.

# Set testing environment variables before importing server
os.environ["TESTING"] = "true"
os.environ["STORAGE_DIR"] = "./data"
os.environ["HF_TOKEN"] = "dummy_hf_token"
os.environ["ELEVENLABS_API_KEY"] = "dummy_elevenlabs"
os.environ["OPENAI_API_KEY"] = "dummy_openai"
os.environ["WORKER_API_KEY"] = "secret_test_key"
os.environ["ENABLE_DRY_RUN"] = "true"

# Add repository root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server

class TestHunyuanVideoAPI(unittest.TestCase):
    def setUp(self):
        server.app.config["TESTING"] = True
        self.client = server.app.test_client()
        # Reset rate limits for testing
        server.RATE_LIMIT_WINDOW.clear()

    def test_unauthorized_access(self):
        """Verify that requests without a valid API key are rejected with 401."""
        response = self.client.post("/render-sequence", json={"topic": "Test"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("error", response.get_json())

        response = self.client.get("/tasks/some-job")
        self.assertEqual(response.status_code, 401)

    def test_authorized_access(self):
        """Verify that requests with a valid API key are authorized."""
        headers = {"X-API-Key": "secret_test_key"}
        response = self.client.post("/render-sequence", json={}, headers=headers)
        # Missing topic should return 400, but not 401
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing or empty 'topic'", response.get_json()["error"])

    def test_topic_length_validation(self):
        """Verify that overly long topics are rejected."""
        headers = {"X-API-Key": "secret_test_key"}
        long_topic = "A" * 201
        response = self.client.post("/render-sequence", json={"topic": long_topic}, headers=headers)
        self.assertEqual(response.status_code, 413)
        self.assertIn("Topic too long", response.get_json()["error"])

    def test_narration_length_validation(self):
        """Verify that overly large narrations are rejected."""
        headers = {"X-API-Key": "secret_test_key"}
        large_narration = {
            "hook": "A" * 1000,
            "problem": "B" * 1000,
            "rozwiązanie": "C" * 1000
        }
        response = self.client.post("/render-sequence", json={
            "topic": "Test",
            "narration": large_narration
        }, headers=headers)
        self.assertEqual(response.status_code, 413)
        self.assertIn("Narration too large", response.get_json()["error"])

    def test_hashtags_validation(self):
        """Verify that hashtags constraints are strictly validated."""
        headers = {"X-API-Key": "secret_test_key"}
        
        # Invalid format (no hash)
        response = self.client.post("/render-sequence", json={
            "topic": "Test",
            "hashtags": ["invalid_tag"]
        }, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("must start with #", response.get_json()["error"])

        # Contains spaces
        response = self.client.post("/render-sequence", json={
            "topic": "Test",
            "hashtags": ["#invalid tag"]
        }, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("cannot contain spaces", response.get_json()["error"])

        # Too many hashtags
        too_many_tags = [f"#tag{i}" for i in range(10)]
        response = self.client.post("/render-sequence", json={
            "topic": "Test",
            "hashtags": too_many_tags
        }, headers=headers)
        self.assertEqual(response.status_code, 413)
        self.assertIn("Too many hashtags", response.get_json()["error"])

    def test_ssrf_webhook_protection(self):
        """Verify that local or private webhook URLs are rejected to protect against SSRF."""
        headers = {"X-API-Key": "secret_test_key"}

        unsafe_urls = [
            "http://localhost/webhook",
            "http://127.0.0.1/webhook",
            "http://169.254.169.254/computeMetadata/v1",
            "http://10.0.0.1/webhook",
            "http://192.168.1.1/webhook",
            "http://172.16.0.1/webhook"
        ]

        for url in unsafe_urls:
            response = self.client.post("/render-sequence", json={
                "topic": "Test",
                "webhookUrl": url
            }, headers=headers)
            self.assertEqual(response.status_code, 400, f"Failed to reject unsafe URL: {url}")
            self.assertIn("unsafe 'webhookUrl'", response.get_json()["error"])

    def test_successful_dry_run_queue(self):
        """Verify that a valid request is successfully queued in dry-run mode."""
        headers = {"X-API-Key": "secret_test_key"}
        response = self.client.post("/render-sequence", json={
            "topic": "Inwestowanie w akcje",
            "webhookUrl": "https://example.com/webhook"
        }, headers=headers)
        self.assertEqual(response.status_code, 202)
        data = response.get_json()
        self.assertEqual(data["status"], "queued")
        self.assertTrue(data["job_id"])
        self.assertTrue(data["status_url"])

if __name__ == "__main__":
    unittest.main()
