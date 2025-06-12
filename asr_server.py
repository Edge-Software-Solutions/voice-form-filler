#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import websockets
import sherpa_onnx


def get_args():
    parser = argparse.ArgumentParser(description="Streaming ASR WebSocket server")
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", required=True, help="Path to decoder.onnx")
    parser.add_argument("--joiner", required=True, help="Path to joiner.onnx")
    parser.add_argument("--tokens", required=True, help="Path to tokens.txt")
    parser.add_argument(
        "--model-type",
        type=str,
        default="nemo_transducer",
        help="Set to 'nemo_transducer' for NVIDIA Parakeet models",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    return parser.parse_args()


def create_recognizer(args):
    for attr in ["encoder", "decoder", "joiner", "tokens"]:
        assert Path(getattr(args, attr)).is_file(), getattr(args, attr)

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        enable_endpoint_detection=False,
        model_type=args.model_type,
    )
    return recognizer


async def stream_handler(websocket, recognizer):
    stream = recognizer.create_stream()
    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                if data.get("type") == "stop":
                    stream.input_finished()
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                        partial = recognizer.get_result(stream)
                        await websocket.send(json.dumps({"type": "realtime", "text": partial}))
                    final = recognizer.get_result(stream)
                    await websocket.send(json.dumps({"type": "fullSentence", "text": final}))
                    recognizer.reset(stream)
                    stream = recognizer.create_stream()
            else:
                pcm = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
                stream.accept_waveform(16000, pcm.tolist())
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                    partial = recognizer.get_result(stream)
                    await websocket.send(json.dumps({"type": "realtime", "text": partial}))
    except websockets.exceptions.ConnectionClosed:
        pass


async def main():
    args = get_args()
    recognizer = create_recognizer(args)
    async with websockets.serve(lambda ws: stream_handler(ws, recognizer), "0.0.0.0", args.port):
        print(f"ASR server listening on ws://localhost:{args.port}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
