#!/usr/bin/env python3
"""Minimal OpenAI-compatible Qwen-Image server for the V2-S3 spike.

The mock backend makes the HTTP contract testable without loading model weights.
The real backend deliberately loads the oversized transformer separately: Diffusers'
pipeline-level ``device_map="balanced"`` only places whole pipeline components and
therefore cannot split Qwen-Image's ~40.9 GB bf16 transformer across 32 GB cards.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MOCK_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
SIZE_RE = re.compile(r"^([1-9][0-9]*)x([1-9][0-9]*)$")


class GenerationRequest(BaseModel):
    model: str
    prompt: str
    n: int = 1
    size: str = "1328x1328"
    response_format: str = "b64_json"
    negative_prompt: str = " "
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    seed: int | None = None


class ImageBackend(Protocol):
    load_seconds: float
    placement: dict[str, Any]

    def generate(self, request: GenerationRequest, width: int, height: int) -> bytes: ...


@dataclass
class MockBackend:
    load_seconds: float = 0.0
    placement: dict[str, Any] = field(default_factory=lambda: {"mode": "mock"})

    def generate(self, request: GenerationRequest, width: int, height: int) -> bytes:
        return MOCK_PNG


class DiffusersBackend:
    def __init__(self, model_path: str, max_memory: dict[int | str, str]) -> None:
        import torch
        from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        started = time.monotonic()
        # Qwen-Image's transformer is larger than either GPU. Dispatch it at the
        # block level; putting device_map="balanced" on the pipeline does not.
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            max_memory=max_memory,
            local_files_only=True,
        )
        # This conservative prototype avoids double-spending the same GPU budgets.
        # A production planning hook must budget all components jointly.
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            os.path.join(model_path, "text_encoder"),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
        )
        self.pipeline = QwenImagePipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder=text_encoder,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.load_seconds = time.monotonic() - started
        self.placement = {
            "mode": "component-balanced-transformer",
            "transformer": transformer.hf_device_map,
            "text_encoder": "cpu",
            "max_memory": {str(key): value for key, value in max_memory.items()},
        }

    def generate(self, request: GenerationRequest, width: int, height: int) -> bytes:
        import torch

        generator = None
        if request.seed is not None:
            # A CPU generator is portable across Accelerate-dispatched modules.
            generator = torch.Generator(device="cpu").manual_seed(request.seed)
        image = self.pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
            true_cfg_scale=request.true_cfg_scale,
            generator=generator,
        ).images[0]
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()


def parse_max_memory(raw: str) -> dict[int | str, str]:
    value = json.loads(raw)
    if not isinstance(value, dict) or not value:
        raise ValueError("max-memory must be a non-empty JSON object")
    result: dict[int | str, str] = {}
    for raw_device, limit in value.items():
        if not isinstance(limit, str) or not limit.strip():
            raise ValueError(f"memory limit for {raw_device!r} must be a non-empty string")
        if raw_device == "cpu":
            device: int | str = raw_device
        elif isinstance(raw_device, str) and raw_device.isdigit():
            device = int(raw_device)
        else:
            raise ValueError(f"device {raw_device!r} must be a logical GPU index or 'cpu'")
        result[device] = limit
    return result


def parse_size(size: str) -> tuple[int, int]:
    match = SIZE_RE.fullmatch(size)
    if match is None:
        raise ValueError("size must have the form WIDTHxHEIGHT")
    width, height = (int(value) for value in match.groups())
    if width > 2048 or height > 2048 or width % 8 or height % 8:
        raise ValueError("width and height must be <= 2048 and divisible by 8")
    return width, height


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "code": code,
            }
        },
    )


def create_app(backend: ImageBackend, served_model: str) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    generation_lock = asyncio.Lock()

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        return error(500, "generation_failed", str(exc))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": served_model,
            "load_seconds": backend.load_seconds,
            "placement": backend.placement,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": served_model, "object": "model", "owned_by": "qwen"}],
        }

    @app.post("/v1/images/generations")
    async def generate(request: GenerationRequest) -> JSONResponse:
        if request.model != served_model:
            return error(404, "model_not_found", f"model {request.model!r} is not served")
        if request.n != 1:
            return error(400, "unsupported_n", "this prototype supports n=1 only")
        if request.response_format != "b64_json":
            return error(
                400,
                "unsupported_response_format",
                "this prototype supports response_format=b64_json only",
            )
        try:
            width, height = parse_size(request.size)
        except ValueError as exc:
            return error(400, "invalid_size", str(exc))
        async with generation_lock:
            image = await asyncio.to_thread(backend.generate, request, width, height)
        return JSONResponse(
            content={
                "created": int(time.time()),
                "data": [{"b64_json": base64.b64encode(image).decode("ascii")}],
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="Qwen/Qwen-Image")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--max-memory",
        default='{"0":"20GiB","1":"28GiB"}',
        help="Accelerate limits for the transformer, using logical CUDA indices",
    )
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    backend: ImageBackend
    if args.mock:
        backend = MockBackend()
    else:
        backend = DiffusersBackend(args.model_path, parse_max_memory(args.max_memory))
    uvicorn.run(create_app(backend, args.served_model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
