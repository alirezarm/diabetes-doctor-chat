import argparse
from pathlib import Path
import gradio as gr
# from lora import generate
import mlx.core as mx
import utils as lora_utils

def build_parser():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    # Generation args
    parser.add_argument(
        "--model",
        default="model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=200,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    return parser


def generate(model, prompt, tokenizer, args):
    print(prompt, end="", flush=True)
    prompt_formatted = """Patient's Query:\n\n {} ###\n\n""".format(prompt)
    prompt = mx.array(tokenizer.encode(prompt_formatted))

    tokens = []
    skip = 0
    for token, n in zip(
        lora_utils.generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)
    return tokenizer.decode(tokens) if len(tokens) != 0 else "No tokens generated for this prompt"


def load(model_path):
    print("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(model_path)
    return model, tokenizer


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model, tokenizer = load(args.model)

    def chat_doctor_response(message, history):
        output = generate(model, message, tokenizer, args)
        return output

    gr.ChatInterface(fn=chat_doctor_response, title="Base LLM").launch()
