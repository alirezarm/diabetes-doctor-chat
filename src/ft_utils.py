import mlx.core as mx
import utils as lora_utils


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
