from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def loadModelAndSaveWeights(path: str) -> None:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading/loading GPT-2...")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    print("Saving weights...")

    with torch.inference_mode():
        for name, param in model.named_parameters():
            values = param.detach().cpu().numpy().flatten()
            output_file = output_dir / f"{name}.txt"

            with open(output_file, "w") as f:
                f.write(" ".join(str(v) for v in values))

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output_dir / "tokenizer")



