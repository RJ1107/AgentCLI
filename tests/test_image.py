from __future__ import annotations

from PIL import Image

from agentcli.image import parse_image_references
from agentcli.llm.openai_compatible import OpenAICompatibleClient


def test_parse_local_image_reference(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(image_path)

    content = parse_image_references(f"look @image:{image_path.name}", str(tmp_path))

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_non_vision_model_omits_image_payload(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), "red").save(image_path)
    content = parse_image_references(f"look @image:{image_path.name}", str(tmp_path))
    client = OpenAICompatibleClient(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://example.com/v1",
    )

    formatted = client._format_content(content)

    assert isinstance(formatted, str)
    assert "Image omitted" in formatted
