import torch
import random
import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import (
    get_activation_addition_input_pre_hook,
    get_all_direction_ablation_hooks,
)

from pipeline.submodules.generate_directions import (
    generate_directions,
    generate_activations_for_harmful,
)
from pipeline.submodules.select_direction import (
    select_direction,
    get_refusal_scores,
    refusal_metric_plot,
)
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


# def parse_arguments():
#     """Parse model path argument from command line."""
#     parser = argparse.ArgumentParser(description="Parse model path argument.")
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
#     return parser.parse_args()
def parse_arguments():
    """Parse model path and optional direction file path from command line."""
    parser = argparse.ArgumentParser(
        description="Parse model path and optional direction file path."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--direction_file", type=str, help="Path to the JSON file containing directions"
    )
    parser.add_argument(
        "--activation_prompts",
        action="store_true",
        help="Activate prompts if this flag is set",
    )
    parser.add_argument(
        "--set_temperature",
        type=float,
        default=1.0,
        help="Temperature for text generation (default: 1.0)",
        nargs="?",
    )

    return parser.parse_args()


def load_directions(file_path):
    """Load direction from a .pt file."""
    return torch.load(file_path, map_location=torch.device("cpu"))


def load_and_sample_datasets_for_activations(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(
        load_dataset_split(harmtype="harmful", split="train", instructions_only=True),
        cfg.n_train_for_activations,
    )
    harmless_train = random.sample(
        load_dataset_split(harmtype="harmless", split="train", instructions_only=True),
        cfg.n_train_for_activations,
    )
    harmful_val = random.sample(
        load_dataset_split(harmtype="harmful", split="val", instructions_only=True),
        cfg.n_val_for_activations,
    )
    harmless_val = random.sample(
        load_dataset_split(harmtype="harmless", split="val", instructions_only=True),
        cfg.n_val_for_activations,
    )
    return harmful_train, harmless_train, harmful_val, harmless_val

def load_and_sample_datasets(cfg):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    random.seed(42)
    harmful_train = random.sample(
        load_dataset_split(harmtype="harmful", split="train", instructions_only=True),
        cfg.n_train,
    )
    harmless_train = random.sample(
        load_dataset_split(harmtype="harmless", split="train", instructions_only=True),
        cfg.n_train,
    )
    harmful_val = random.sample(
        load_dataset_split(harmtype="harmful", split="val", instructions_only=True),
        cfg.n_val,
    )
    harmless_val = random.sample(
        load_dataset_split(harmtype="harmless", split="val", instructions_only=True),
        cfg.n_val,
    )
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(
    cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val
):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [
            inst
            for inst, score in zip(dataset, scores.tolist())
            if comparison(score, threshold)
        ]

    if cfg.filter_train:
        harmful_train_scores = get_refusal_scores(
            model_base.model,
            harmful_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmless_train_scores = get_refusal_scores(
            model_base.model,
            harmless_train,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmful_train = filter_examples(
            harmful_train, harmful_train_scores, 0, lambda x, y: x > y
        )
        harmless_train = filter_examples(
            harmless_train, harmless_train_scores, 0, lambda x, y: x < y
        )

    if cfg.filter_val:
        harmful_val_scores = get_refusal_scores(
            model_base.model,
            harmful_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmless_val_scores = get_refusal_scores(
            model_base.model,
            harmless_val,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
        )
        harmful_val = filter_examples(
            harmful_val, harmful_val_scores, 0, lambda x, y: x > y
        )
        harmless_val = filter_examples(
            harmless_val, harmless_val_scores, 0, lambda x, y: x < y
        )

    return harmful_train, harmless_train, harmful_val, harmless_val


def generate_and_save_candidate_directions(
    cfg, model_base, harmful_train, harmless_train
):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), "generate_directions")):
        os.makedirs(os.path.join(cfg.artifact_path(), "generate_directions"))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"),
    )

    torch.save(
        mean_diffs,
        os.path.join(cfg.artifact_path(), "generate_directions/mean_diffs.pt"),
    )

    return mean_diffs


def normalize_vector(vector,model_base):
    """Normalize a given vector to have unit length."""
    # if dim>1, so we normalize the multi-dimensional tensor input along the last dimension
    if vector.dim() > 1:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    else:
        # if its single-dimensional tensor, we just take the norm
        assert vector.shape == (model_base.model.config.hidden_size,), "Shape of the 1D vector does not match expected (x,)"
        norm = torch.linalg.vector_norm(vector, dim=0, keepdim=True)
   
    if torch.isclose(norm, torch.tensor(0.,dtype=norm.dtype, device=norm.device)).any():
        return vector
    return vector / norm


def plotting_refusal_with_activations(
    layers, average_dot_products, resultant_dot_products, activations_dir
):
    layers=range(layers)
    plt.figure(figsize=(10, 5))  # Set the figure size as needed

    # Plotting Average Dot Products
    plt.plot(layers, average_dot_products, label="mean(similarity(A_i, R))", marker="o")

    # Plotting Resultant Dot Products
    plt.plot(layers, resultant_dot_products, label="similarity(mean(A_i), R)", marker="o")

    plt.title("Dot Products Comparison Across Layers")  # Title of the plot
    plt.xlabel("Layer Number")  # Label for the x-axis
    plt.ylabel("Dot Product Value")  # Label for the y-axis
    plt.xticks(layers)  # Ensure all layer numbers are marked
    plt.legend()

    plot_file_path = os.path.join(activations_dir, "dot_products_plot.png")
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Plot saved to {plot_file_path}")


def computing_dot_product(
    model_base,
    layer_wise_activations: List[torch.Tensor],
    refusal_direction: torch.Tensor,
    average_dot_products: List[torch.Tensor],
    resultant_dot_products: List[torch.Tensor],
    activations_dir: str, 
    layers: int, 
    harmful_train: List[torch.Tensor]
) -> None:

    device=refusal_direction.device
    #layer_wise_activations is list, converting it to tensors for matching shape
    layer_wise_activations_tensor = torch.stack(layer_wise_activations).to(device)
    assert layer_wise_activations_tensor.shape == (len(harmful_train), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    normalize_vector(layer_wise_activations_tensor)

    #Getting mean(cos(A_i, R))
    average_dot_products = torch.matmul(layer_wise_activations_tensor, refusal_direction)  # Shape: [samples, n_layer]
    assert average_dot_products.shape == (len(harmful_train),model_base.model.config.num_hidden_layers)
    average_dot_products = dot_products.mean(dim=0) 
    average_dot_products=average_dot_products.tolist()

    #getting cos(mean(A_i), R)
    resultant_vectors = layer_wise_activations_tensor.sum(dim=0) # shape n_layers, d_model
    normalize_vector(resultant_vectors)
    assert average_dot_products.shape == (model_base.model.config.num_hidden_layers,model_base.model.config.hidden_size)
    resultant_vectors = torch.matmul(resultant_vectors, refusal_direction)
    resultant_vectors= resultant_vectors.tolist()

    # Create and display a DataFrame with results
    data = {
        "Layer": list(range(layers)),
        "mean(cos(A_i, R))": average_dot_products,
        "cos(mean(A_i), R)": resultant_dot_products,
    }
    df = pd.DataFrame(data)

    # Save average_dot_products
    average_activation_file_path = os.path.join(
        activations_dir, "average_dot_products.npy"
    )
    np.save(average_activation_file_path, average_dot_products)

    # Save resultant_dot_products
    resultant_activation_file_path = os.path.join(
        activations_dir, "resultant_dot_products.npy"
    )
    np.save(resultant_activation_file_path, resultant_dot_products)

    # Save DataFrame as CSV
    df_file_path = os.path.join(activations_dir, "cosine_similarity_results.csv")
    df.to_csv(df_file_path, index=False)

    print("Data saved successfully.")
    print(df)

    return average_dot_products, resultant_dot_products


def generate_and_save_activations(cfg, model_base, harmful_train, refusal_direction):
    """Generate and save candidate directions."""
    activations_dir = os.path.join(cfg.artifact_path(), "generate_activations")

    # Create the directory if it doesn't exist
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)

    # for storing cos with refusal_direction
    layer_wise_activations = []

    for i in range(len(harmful_train)):
        mean_activations = generate_activations_for_harmful(
            model_base,
            harmful_train[i],
        )
        file_path = os.path.join(activations_dir, f"mean_activations_for_{i}_prompt.pt")
        assert mean_activations.shape == (5, model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
        # mean activations has a size of 5*18*d_model -> it takes activations of last 5 positions of prompt
        layer_wise_activations.append(mean_activations[-1, :, :])
        torch.save(mean_activations, file_path)

    refusal_direction = normalize_vector(refusal_direction,model_base)
    average_dot_products = []
    resultant_dot_products = []
    layers = model_base.model.config.num_hidden_layers

    average_dot_products, resultant_dot_products = computing_dot_product(
        model_base,
        layer_wise_activations,
        refusal_direction,
        average_dot_products,
        resultant_dot_products,
        activations_dir,
        layers,
        harmful_train,
    )
    plotting_refusal_with_activations(
        layers, average_dot_products, resultant_dot_products, activations_dir
    )


def select_and_save_direction(
    cfg, model_base, harmful_val, harmless_val, candidate_directions
):
    """Select and save the direction."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), "select_direction")):
        os.makedirs(os.path.join(cfg.artifact_path(), "select_direction"))

    pos, layer, direction = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"),
    )

    with open(f"{cfg.artifact_path()}/direction_metadata.json", "w") as f:
        json.dump({"pos": pos, "layer": layer}, f, indent=4)

    torch.save(direction, f"{cfg.artifact_path()}/direction.pt")

    return pos, layer, direction


def generate_and_save_completions_for_dataset(
    cfg,
    model_base,
    fwd_pre_hooks,
    fwd_hooks,
    intervention_label,
    dataset_name,
    temperature=1.0,
    dataset=None,
):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), "completions")):
        os.makedirs(os.path.join(cfg.artifact_path(), "completions"))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(
        dataset,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=cfg.max_new_tokens,
        temperature=temperature,
    )

    with open(
        f"{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json",
        "w",
    ) as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(
    cfg, intervention_label, dataset_name, eval_methodologies
):
    """Evaluate completions and save results for a dataset."""
    with open(
        os.path.join(
            cfg.artifact_path(),
            f"completions/{dataset_name}_{intervention_label}_completions.json",
        ),
        "r",
    ) as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(
            cfg.artifact_path(),
            "completions",
            f"{dataset_name}_{intervention_label}_evaluations.json",
        ),
    )

    with open(
        f"{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json",
        "w",
    ) as f:
        json.dump(evaluation, f, indent=4)


def evaluate_loss_for_datasets(
    cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label
):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), "loss_evals")):
        os.makedirs(os.path.join(cfg.artifact_path(), "loss_evals"))

    on_distribution_completions_file_path = os.path.join(
        cfg.artifact_path(), "completions/harmless_baseline_completions.json"
    )

    loss_evals = evaluate_loss(
        model_base,
        fwd_pre_hooks,
        fwd_hooks,
        batch_size=cfg.ce_loss_batch_size,
        n_batches=cfg.ce_loss_n_batches,
        completions_file_path=on_distribution_completions_file_path,
    )

    with open(
        f"{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json", "w"
    ) as f:
        json.dump(loss_evals, f, indent=4)


def run_pipeline(model_path, refusal_direction=None, activation_prompts=False,temperature=1.0):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    # To get activations of each layer
    if activation_prompts:
        harmful_train, harmless_train, harmful_val, harmless_val = (
            load_and_sample_datasets_for_activations(cfg)
        )
        generate_and_save_activations(cfg, model_base, harmful_train, refusal_direction)
        print("Loaded activations for each layer")

    # Load and sample datasets
    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(
        cfg
    )

    if refusal_direction is not None:
        refusal_metric_plot(
            cfg, model_base, harmful_val, harmless_val, refusal_direction
        )
    else:
    # 1. Generate candidate refusal directions
        candidate_directions = generate_and_save_candidate_directions(
        cfg, model_base, harmful_train, harmless_train
    )
    # 2. Select the most effective refusal direction
        pos, layer, refusal_direction = select_and_save_direction(
            cfg, model_base, harmful_val, harmless_val, candidate_directions
        )

    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    harmful_dataset_name = cfg.evaluation_datasets[0]

    harmful_test = random.sample(
        load_dataset(harmful_dataset_name), cfg.n_test
    )
    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        baseline_fwd_pre_hooks,
        baseline_fwd_hooks,
        "baseline",
        harmful_dataset_name,
        temperature,
        dataset=harmful_test,
    )

    harmless_test = random.sample(
        load_dataset_split(harmtype="harmless", split="test"),
        cfg.n_test,
    )
    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        baseline_fwd_pre_hooks,
        baseline_fwd_hooks,
        "baseline",
        "harmless",
        temperature,
        dataset=harmless_test,
    )

    for layer in range(model_base.model.config.num_hidden_layers):
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(
            model_base, refusal_direction,layer
        )

        actadd_fwd_pre_hooks, actadd_fwd_hooks = (
            [
                (
                    model_base.model_block_modules[layer],
                    get_activation_addition_input_pre_hook(
                        vector=refusal_direction, coeff=-1.0
                    ),
                )
            ],
            [],
        )

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            ablation_fwd_pre_hooks,
            ablation_fwd_hooks,
            f"ablation_{layer}",
            harmful_dataset_name,
            temperature,
            harmful_test,
        )
        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            actadd_fwd_pre_hooks,
            actadd_fwd_hooks,
            f"actadd_{layer}",
            harmful_dataset_name,
            temperature,
            harmful_test,
        )

        # 4a. Generate and save completions on harmless evaluation dataset
        actadd_refusal_pre_hooks, actadd_refusal_hooks = (
            [
                (
                    model_base.model_block_modules[layer],
                    get_activation_addition_input_pre_hook(
                        vector=refusal_direction, coeff=+1.0
                    ),
                )
            ],
            [],
        )

        generate_and_save_completions_for_dataset(
            cfg,
            model_base,
            actadd_refusal_pre_hooks,
            actadd_refusal_hooks,
            f"actadd_{layer}",
            "harmless",
            temperature,
            dataset=harmless_test,
        )

if __name__ == "__main__":
    args = parse_arguments()
    directions = load_directions(args.direction_file) if args.direction_file else None
    run_pipeline(
        model_path=args.model_path,
        refusal_direction=directions,
        activation_prompts=args.activation_prompts,
        temperature=args.set_temperature,
    )
