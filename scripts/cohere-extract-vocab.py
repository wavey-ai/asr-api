#!/usr/bin/env python3
"""Extract Cohere tokenizer.model pieces into vocab.json for the MLX backend."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract vocab.json from tokenizer.model")
    parser.add_argument("--model-dir", required=True, type=Path, help="Cohere model directory")
    args = parser.parse_args()

    model_path = args.model_dir / "tokenizer.model"
    output_path = args.model_dir / "vocab.json"
    if not model_path.exists():
        raise FileNotFoundError(f"tokenizer.model not found at {model_path}")

    try:
        import sentencepiece as spm
    except ImportError as error:
        raise ImportError("sentencepiece is required: pip install sentencepiece") from error

    processor = spm.SentencePieceProcessor()
    processor.Load(str(model_path))
    vocab = {str(idx): processor.IdToPiece(idx) for idx in range(processor.GetPieceSize())}
    output_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(vocab)} tokens to {output_path}")


if __name__ == "__main__":
    main()
