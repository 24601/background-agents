"""Tests for OpenCodeServer.build_provider_overrides().

The sandbox routes a provider through an OpenAI-compatible gateway by setting
that provider's *_BASE_URL alongside its API key. These cases pin the mapping
from environment to OpenCode's `provider` config block, including the
must-stay-absent case: a provider without an override contributes nothing, so
OpenCode keeps its models.dev-derived endpoint.
"""

from sandbox_runtime.opencode_server import OpenCodeServer


class TestBuildProviderOverrides:
    def test_no_base_urls_yields_no_provider_block(self):
        assert OpenCodeServer.build_provider_overrides({}) == {}

    def test_blank_base_url_is_ignored(self):
        assert OpenCodeServer.build_provider_overrides({"OPENAI_BASE_URL": "   "}) == {}

    def test_openai_base_url_becomes_provider_options(self):
        overrides = OpenCodeServer.build_provider_overrides(
            {"OPENAI_BASE_URL": "https://gateway.example.com/v1"}
        )
        assert overrides == {"openai": {"options": {"baseURL": "https://gateway.example.com/v1"}}}

    def test_only_overridden_providers_appear(self):
        overrides = OpenCodeServer.build_provider_overrides(
            {"OPENAI_BASE_URL": "https://gateway.example.com/v1", "ANTHROPIC_API_KEY": "sk-x"}
        )
        assert set(overrides) == {"openai"}

    def test_multiple_providers_are_independent(self):
        overrides = OpenCodeServer.build_provider_overrides(
            {
                "OPENAI_BASE_URL": "https://gateway.example.com/v1",
                "XAI_BASE_URL": "https://gateway.example.com/xai/v1",
            }
        )
        assert overrides == {
            "openai": {"options": {"baseURL": "https://gateway.example.com/v1"}},
            "xai": {"options": {"baseURL": "https://gateway.example.com/xai/v1"}},
        }

    def test_surrounding_whitespace_is_trimmed(self):
        overrides = OpenCodeServer.build_provider_overrides(
            {"OPENAI_BASE_URL": "  https://gateway.example.com/v1  "}
        )
        assert overrides["openai"]["options"]["baseURL"] == "https://gateway.example.com/v1"
