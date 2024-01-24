import argparse
from pathlib import Path
import gradio as gr
import mlx.core as mx
import utils as lora_utils
from ft_utils import generate

def build_parser():
    parser = argparse.ArgumentParser(description="Gradio Chatbot for Fine-Tuned Model.")
    # Generation args
    parser.add_argument(
        "--model",
        default="model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="adapters.npz",
        help="Save/load path for the trained adapter weights.",
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


def load(model_path, adapter_file):
    print("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(model_path)

    print("Loading LoRA adapter weights")
    if not Path(adapter_file).is_file():
        raise ValueError(
            f"Adapter file {adapter_file} missing. "
            "Use --train to learn and save the adapters.npz."
        )
    model.load_weights(adapter_file, strict=False)
    return model, tokenizer


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    model, tokenizer = load(args.model, args.adapter_file)

    def chat_doctor_response(message, history):
        output = generate(model, message, tokenizer, args)
        return output
    gr.ChatInterface(
        fn=chat_doctor_response,
        title="Dr. Diabetes"
    ).launch()
