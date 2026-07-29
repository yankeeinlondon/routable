import base64
import unittest

from fastapi.testclient import TestClient

from server import MOCK_PNG, MockBackend, create_app, parse_max_memory, parse_size


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app(MockBackend(), "Qwen/Qwen-Image"))

    def test_runner_readiness_contract(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["id"], "Qwen/Qwen-Image")

    def test_image_generation_contract(self) -> None:
        response = self.client.post(
            "/v1/images/generations",
            json={
                "model": "Qwen/Qwen-Image",
                "prompt": "contract probe",
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(base64.b64decode(response.json()["data"][0]["b64_json"]), MOCK_PNG)

    def test_wrong_model_and_unsupported_output_fail_before_generation(self) -> None:
        wrong = self.client.post(
            "/v1/images/generations",
            json={"model": "other", "prompt": "must not run"},
        )
        self.assertEqual(wrong.status_code, 404)
        self.assertEqual(wrong.json()["error"]["code"], "model_not_found")
        output = self.client.post(
            "/v1/images/generations",
            json={
                "model": "Qwen/Qwen-Image",
                "prompt": "must not run",
                "response_format": "url",
            },
        )
        self.assertEqual(output.status_code, 400)
        self.assertEqual(output.json()["error"]["code"], "unsupported_response_format")


class ConfigurationTests(unittest.TestCase):
    def test_asymmetric_gpu_limits_are_normalized(self) -> None:
        self.assertEqual(
            parse_max_memory('{"0":"8GiB","1":"30GiB","cpu":"96GiB"}'),
            {0: "8GiB", 1: "30GiB", "cpu": "96GiB"},
        )

    def test_image_size_is_bounded(self) -> None:
        self.assertEqual(parse_size("1328x1328"), (1328, 1328))
        for invalid in ("1328", "1329x1328", "4096x4096"):
            with self.assertRaises(ValueError):
                parse_size(invalid)


if __name__ == "__main__":
    unittest.main()
