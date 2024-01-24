import argparse
from pathlib import Path
import mlx.core as mx
import utils as lora_utils
from lora import Dataset
import evaluate
from tqdm import tqdm


rouge_score = evaluate.load("rouge")


def build_parser():
    parser = argparse.ArgumentParser(description="Rouge Evaluation.")
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
        "--data",
        type=str,
        default="data/",
        help="Directory with test.jsonl file",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=300,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    return parser


def load_base_model(model_path):
    print("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(model_path)
    return model, tokenizer


def load_ft_model(model_path, adapter_file):
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


def generate(model, prompt, tokenizer, temp=0.8, max_tokens=200):
    """Generate response for a prompt"""
    prompt_formatted = """Patient's Query:\n\n {} ###\n\n""".format(prompt)
    prompt = mx.array(tokenizer.encode(prompt_formatted))

    tokens = []
    for token, _ in zip(
        lora_utils.generate(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
    return tokenizer.decode(tokens) if len(tokens) != 0 else "No tokens generated for this prompt"


def generate_responses(tokenizer, model, questions, temp=0.8, max_tokens=200):
    responses = []
    for question in tqdm(questions):
        responses.append(
            generate(model, question, tokenizer, temp=temp, max_tokens=max_tokens)
        )
    return responses


def compute_rouge_score(generated, reference):
    return rouge_score.compute(
        predictions=generated,
        references=reference,
        use_stemmer=True,
        
    )


def dataset_to_prompt_completions(data_folder):
    test = Dataset(Path(data_folder) / "test.jsonl")
    prompts = []
    completions = []
    for prompt_completion in test:
        prompt = prompt_completion.partition("### Patient query: ")[2].partition("### Doctor opinion")[0]
        completion = prompt_completion.partition("### Doctor opinion: ")[2]
        prompts.append(prompt)
        completions.append(completion)
    return prompts, completions


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    # load base and fine-tuned models
    base_model, base_tokenizer = load_base_model(args.model)
    ft_model, ft_tokenizer = load_ft_model(args.model, args.adapter_file)
    # convert test dataset to prompt and completion lists
    prompts, completions = dataset_to_prompt_completions(args.data)
    # generate model responses
    print("[INFO] generate responses from base model")
    base_model_responses = generate_responses(
        base_tokenizer, base_model, prompts, args.temp, args.max_tokens
    )
    print("[INFO] generate responses from fine-tuned model")
    ft_model_responses = generate_responses(
        ft_tokenizer, ft_model, prompts, args.temp, args.max_tokens
    )
    # compute rouge
    print("[INFO] computing base model rouge")
    base_rouge = compute_rouge_score(base_model_responses, completions)
    print("[INFO] computing fine-tuned model rouge")
    ft_rouge = compute_rouge_score(ft_model_responses, completions)

    print("[INFO] base model rouge")
    print(base_rouge)
    print("[INFO] fine-tuned model rouge")
    print(ft_rouge)
