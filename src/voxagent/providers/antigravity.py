"""Antigravity provider for voxagent.

Provides access to Google Gemini models via OAuth credentials from the
Gemini CLI (~/.gemini/oauth_creds.json).
"""

from __future__ import annotations

import json
import os
import platform
import secrets
import time
import uuid
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

import httpx
from voxagent.providers.base import (
    BaseProvider,
    ErrorChunk,
    MessageEndChunk,
    ProviderRequestChunk,
    StreamChunk,
    TextDeltaChunk,
    ToolUseChunk,
)
from voxagent.types.messages import Message, ToolCall, ToolResultBlock

if TYPE_CHECKING:
    from voxagent.providers.base import AbortSignal

logger = logging.getLogger(__name__)


class AntigravityProvider(BaseProvider):
    """voxagent provider for Antigravity (Gemini via OAuth).

    Uses Google's cloudcode-pa.googleapis.com API with OAuth credentials
    from the Gemini CLI (~/.gemini/oauth_creds.json).
    """

    OAUTH_FILE = "~/.gemini/oauth_creds.json"
    BASE_URL = "https://cloudcode-pa.googleapis.com/v1internal"
    CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
    CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
    TOKEN_REFRESH_URL = "https://oauth2.googleapis.com/token"
    CLI_VERSION = "0.28.2"

    SUPPORTED_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gemini-2.5-flash",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model

        # Load OAuth credentials
        oauth_path = Path(self.OAUTH_FILE).expanduser()
        if not oauth_path.exists():
            raise FileNotFoundError(
                f"OAuth credentials not found: {oauth_path}. "
                "Run 'gemini login' to authenticate."
            )

        self._oauth_path = oauth_path
        self._oauth_creds = json.loads(oauth_path.read_text())
        self._access_token = self._oauth_creds.get("access_token", "")
        self._project: str | None = None

    @property
    def name(self) -> str:
        return "antigravity"

    @property
    def models(self) -> list[str]:
        return self.SUPPORTED_MODELS.copy()

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def context_limit(self) -> int:
        return 1000000

    def _is_token_expired(self) -> bool:
        expiry = self._oauth_creds.get("expiry_date", 0)
        return time.time() * 1000 >= expiry

    async def _refresh_token_if_needed(self) -> None:
        if not self._is_token_expired():
            return

        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
            response = await client.post(
                self.TOKEN_REFRESH_URL,
                data={
                    "client_id": self.CLIENT_ID,
                    "client_secret": self.CLIENT_SECRET,
                    "refresh_token": self._oauth_creds["refresh_token"],
                    "grant_type": "refresh_token",
                },
            )

        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get(
                "error_description", error_data.get("error", str(error_data))
            )
            raise Exception(f"OAuth token refresh failed: {error_msg}")

        data = response.json()
        self._oauth_creds["access_token"] = data["access_token"]
        self._oauth_creds["expiry_date"] = (
            time.time() * 1000 + data["expires_in"] * 1000 - 60000
        )
        self._access_token = data["access_token"]
        self._oauth_path.write_text(json.dumps(self._oauth_creds, indent=2))

    async def _load_project(self) -> None:
        if self._project:
            return

        headers = self._get_headers()
        body = {
            "metadata": {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        }

        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            response = await client.post(
                f"{self.BASE_URL}:loadCodeAssist",
                json=body,
                headers=headers,
            )

        if response.status_code == 200:
            data = response.json()
            self._project = data.get("cloudaicompanionProject")

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "User-Agent": (
                f"GeminiCLI/{self.CLI_VERSION}/{self._model} "
                f"({platform.system().lower()}; {platform.machine()}) "
                "google-api-nodejs-client/9.15.1"
            ),
            "x-goog-api-client": "gl-node/25.6.1",
            "Accept": "application/json",
        }

    def _convert_messages_to_gemini(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        contents: list[dict[str, Any]] = []
        system_instruction: dict[str, Any] | None = None

        for msg in messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    system_instruction = {"parts": [{"text": msg.content}]}
                continue

            role = "model" if msg.role == "assistant" else "user"
            parts: list[dict[str, Any]] = []

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.params,
                        }
                    })
            elif isinstance(msg.content, str):
                if msg.content:
                    parts.append({"text": msg.content})
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolResultBlock):
                        parts.append({
                            "functionResponse": {
                                "name": block.tool_name or "unknown",
                                "response": {
                                    "result": block.content,
                                    "is_error": block.is_error,
                                }
                            }
                        })
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            parts.append({
                                "functionResponse": {
                                    "name": block.get("tool_name", "unknown"),
                                    "response": {
                                        "result": block.get("content", ""),
                                        "is_error": block.get("is_error", False),
                                    }
                                }
                            })
                        elif "text" in block:
                            parts.append({"text": block["text"]})
                    elif hasattr(block, "text"):
                        parts.append({"text": block.text})
                    elif isinstance(block, str):
                        parts.append({"text": block})

            if parts:
                contents.append({"role": role, "parts": parts})

        return contents, system_instruction

    def _convert_tools_to_gemini(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func_info = tool["function"]
                func_decl: dict[str, Any] = {
                    "name": func_info.get("name", ""),
                    "description": func_info.get("description", ""),
                }
                if "parameters" in func_info:
                    func_decl["parameters"] = func_info["parameters"]
                function_declarations.append(func_decl)
            else:
                func_decl = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                if "parameters" in tool:
                    func_decl["parameters"] = tool["parameters"]
                function_declarations.append(func_decl)

        return [{"functionDeclarations": function_declarations}]

    def _build_request_body(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        contents, system_instruction = self._convert_messages_to_gemini(messages)
        request: dict[str, Any] = {"contents": contents}

        if system:
            request["system_instruction"] = {"parts": [{"text": system}]}
        elif system_instruction:
            request["system_instruction"] = system_instruction

        if tools:
            request["tools"] = self._convert_tools_to_gemini(tools)

        return {
            "model": self._model,
            "project": self._project,
            "user_prompt_id": secrets.token_hex(7),
            "request": request,
        }

    async def stream(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
        abort_signal: "AbortSignal | None" = None,
    ) -> AsyncIterator[StreamChunk]:
        try:
            await self._refresh_token_if_needed()
            await self._load_project()

            body = self._build_request_body(messages, system=system, tools=tools)
            yield ProviderRequestChunk(body=body)

            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}:streamGenerateContent?alt=sse",
                    json=body,
                    headers=self._get_headers(),
                ) as response:
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        try:
                            error_json = json.loads(error_body)
                            error_msg = error_json.get("error", {}).get(
                                "message", error_body.decode()
                            )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            error_msg = error_body.decode() if error_body else "Unknown error"
                        yield ErrorChunk(error=f"Antigravity API error: {error_msg}")
                        return

                    async for line in response.aiter_lines():
                        if abort_signal and abort_signal.aborted:
                            break
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if not data_str:
                            continue
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        response_data = data.get("response", data)
                        candidates = response_data.get("candidates", [])
                        if not candidates:
                            continue

                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])

                        for part in parts:
                            if part.get("thought"): continue
                            if "text" in part:
                                yield TextDeltaChunk(delta=part["text"])
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                yield ToolUseChunk(
                                    tool_call=ToolCall(
                                        id=str(uuid.uuid4()),
                                        name=fc.get("name", ""),
                                        params=fc.get("args", {}),
                                    )
                                )
            yield MessageEndChunk()
        except Exception as e:
            yield ErrorChunk(error=str(e))

    async def complete(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[Any] | None = None,
    ) -> Message:
        await self._refresh_token_if_needed()
        await self._load_project()
        body = self._build_request_body(messages, system=system, tools=tools)

        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            response = await client.post(
                f"{self.BASE_URL}:generateContent",
                json=body,
                headers=self._get_headers(),
            )

        if response.status_code != 200:
            error = response.json().get("error", {})
            raise Exception(f"Antigravity API error: {error.get('message', response.text)}")

        data = response.json()
        response_data = data.get("response", data)
        candidates = response_data.get("candidates", [])

        text_content = ""
        tool_calls: list[ToolCall] = []

        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                if part.get("thought"): continue
                if "text" in part:
                    text_content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append(ToolCall(
                        id=str(uuid.uuid4()),
                        name=fc.get("name", ""),
                        params=fc.get("args", {}),
                    ))

        return Message(role="assistant", content=text_content, tool_calls=tool_calls if tool_calls else None)

    def count_tokens(self, messages: list[Message], system: str | None = None) -> int:
        total_chars = 0
        if system: total_chars += len(system)
        for msg in messages:
            if msg.content: total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.name or "") + len(json.dumps(tc.params or {}))
        return max(1, total_chars // 4)

__all__ = ["AntigravityProvider"]
