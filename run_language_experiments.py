import os
from data_generators import get_tiny_shakespeare_data

# Import the new harness and model factories
from language_experiment_harness import (
    run_comparison,
    create_standard_net_text,
    create_paradox_net_text,
    create_transformer_net_text
)

# --- Experiment Configuration ---

# 1. Define the data generator for the experiment
#    The key is the dataset name, used for saving results.
data_generators = {
    "tiny_shakespeare": get_tiny_shakespeare_data,
}

# 2. Define the models to compare.
#    The key is the model name, used for saving results.
#    The value is the factory function that creates the model.
model_factories = {
    "standard": create_standard_net_text,
    "paradox": create_paradox_net_text,
    "transformer": create_transformer_net_text
}

# 3. Set experiment parameters
N_SEEDS = 5  # Number of times to run each experiment with a different random seed
EPOCHS = 50 # Number of training epochs for each run
SAVE_PATH = "experiment_results_language" # Directory to save results

# --- Run Experiment ---

def main():
    """Main function to run the language model comparison."""
    # Create the output directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    print("Starting language model comparison...")
    
    # Run the full comparison
    run_comparison(
        data_generators=data_generators,
        model_factories=model_factories,
        n_seeds=N_SEEDS,
        epochs=EPOCHS,
        save_path=SAVE_PATH,
        is_classification=True  # Crucial flag for this task
    )
    
    print("\nExperiment comparison finished.")
    print(f"Results have been saved in the '{SAVE_PATH}' directory.")

if __name__ == "__main__":
    main()
