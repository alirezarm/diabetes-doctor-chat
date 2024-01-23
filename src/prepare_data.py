import pandas as pd
import json
import argparse
from datasets import load_dataset, DatasetDict


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare Datasets.")
    parser.add_argument(
        "--test-size",
        default=0.3,
        type=float,
        help="Size of test dataset (equally divided into test and validation).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of sample to use for train/test/validation",
    )
    
    return parser


def generate_prompt(data_point):
    text = f"""### The following is a doctor's opinion on a patient's query:
        \n### Patient query: {data_point['input']}
        \n### Doctor opinion: {data_point['output']}"""
    return {"text": text}


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    df = pd.read_csv("data/care_magic_diabetes.csv")

    # convert to list of dict
    dataset_data = [
        {
            "instruction": row_dict["instruction"],
            "input": row_dict["input"],
            "output": row_dict["output"]
        }
        for row_dict in df.to_dict(orient="records")
    ]

    with open("data/care_magic_diabetes.json", "w") as f:
        json.dump(dataset_data, f)

    data = load_dataset("json", data_files="data/care_magic_diabetes.json")
    

    dataset = data["train"].shuffle().select(range(args.sample_size)).map(generate_prompt)
    dataset = dataset.remove_columns(["instruction", "input", "output"])

    # 70% train, 30% test + validation
    train_test_valid = dataset.train_test_split(test_size=args.test_size)
    
    # Split the 10% test + valid in half test, half valid
    test_valid = train_test_valid["test"].train_test_split(test_size=0.5)

    datasets = DatasetDict({
        "train": train_test_valid["train"],
        "test": test_valid["test"],
        "val": test_valid["train"]}
    )

    print(datasets["train"])
    print(datasets["test"])
    print(datasets["val"])

    names = ("train", "val", "test")
    for name in names:
        with open(f"data/{name}.jsonl", "w") as fid:
            for e in datasets[name]:
                json.dump(e, fid)
                fid.write("\n")
