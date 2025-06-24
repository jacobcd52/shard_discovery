import os
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

def parse_experiment_name(dir_name):
    """Parses the experiment parameters from a directory name."""
    pattern = re.compile(
        r"finegrained_steer_t(\w+)_a(\w+)_alpha(-?[\d\.]+)_layers_([\w\._]+)_finetuning_steps_(\d+)"
    )
    match = pattern.match(dir_name)
    if not match:
        return None
    
    towards, away, alpha, layers, steps = match.groups()
    
    return {
        'towards': towards,
        'away': away,
        'alpha': float(alpha),
        'layers': layers.replace('_', '.'),
        'steps': int(steps),
        'path': dir_name
    }

def find_image(exp_dir, image_name):
    """Finds an image file in the experiment directory."""
    path = os.path.join(exp_dir, image_name)
    return path if os.path.exists(path) else None

def generate_individual_summary_plots():
    """Generates a summary plot for each steering experiment in its directory."""
    results_dir = 'steering_results'
    experiments = []
    for dir_name in os.listdir(results_dir):
        full_path = os.path.join(results_dir, dir_name)
        if os.path.isdir(full_path):
            parsed = parse_experiment_name(dir_name)
            if parsed:
                experiments.append(parsed)

    if not experiments:
        print("No experiments found to plot.")
        return

    for exp in experiments:
        exp_path = os.path.join(results_dir, exp['path'])
        
        # --- Image Paths ---
        dist_towards_img = find_image(exp_path, 'finetuned_model_dist_towards.png')
        dist_away_img = find_image(exp_path, 'finetuned_model_dist_away.png')
        steered_dist_img = find_image(exp_path, 'post_steering_distribution.png')
        diff_img = find_image(exp_path, 'prediction_distribution_difference.png')

        # Skip if essential plots are missing
        if not all([dist_towards_img, dist_away_img, steered_dist_img, diff_img]):
            print(f"Skipping {exp['path']}, not all result images were found.")
            continue

        # --- Create the Individual Plot ---
        fig, axs = plt.subplots(2, 2, figsize=(20, 14))
        
        title = (
            f"Summary: Towards={exp['towards']}, Away={exp['away']}, Alpha={exp['alpha']}\n"
            f"Steps = {exp['steps']} / 64, layers steered with = {exp['layers']}"
        )
        fig.suptitle(
            title,
            fontsize=16, weight='bold'
        )

        # Plot 1: Fine-tuned Towards Model
        axs[0, 0].imshow(mpimg.imread(dist_towards_img))
        axs[0, 0].set_title("Dist: Fine-tuned 'Towards' Model")
        axs[0, 0].axis('off')

        # Plot 2: Fine-tuned Away Model
        axs[0, 1].imshow(mpimg.imread(dist_away_img))
        axs[0, 1].set_title("Dist: Fine-tuned 'Away' Model")
        axs[0, 1].axis('off')

        # Plot 3: Post-Steering Distribution
        axs[1, 0].imshow(mpimg.imread(steered_dist_img))
        axs[1, 0].set_title("Dist: Final Steered Model")
        axs[1, 0].axis('off')
            
        # Plot 4: Prediction Difference
        axs[1, 1].imshow(mpimg.imread(diff_img))
        axs[1, 1].set_title("Change in Predictions")
        axs[1, 1].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the final plot in the experiment's directory
        save_path = os.path.join(exp_path, "_summary_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved summary plot to {save_path}")

if __name__ == '__main__':
    generate_individual_summary_plots() 