import argparse
import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from transformers import BitsAndBytesConfig

from joint_schema_model import load_release_model, systemone


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(8)
    app.state.model, app.state.processor = load_release_model(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        ),
        attn_implementation="sdpa",
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/systemone")
async def decide(request: dict):
    async with inference_lock:
        try:
            return await asyncio.to_thread(
                systemone, app.state.model, app.state.processor, request
            )
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error


uvicorn.run(app, host="127.0.0.1", port=args.port)
