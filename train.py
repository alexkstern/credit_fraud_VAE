import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from model_VAE import VariationalAutoEncoder, vae_loss_function
from data import preprocess, CustomDataset
from tqdm import tqdm

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_reconstruction_error(model, data_loader):
    model.eval()
    total_error = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating reconstruction error", leave=False):
            reconstructed_batch, _, _ = model(batch)
            error = torch.mean((reconstructed_batch - batch) ** 2, dim=1)
            total_error += error.sum().item()
    return total_error / len(data_loader.dataset)

def train_vae(config):
    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])

    # Load and preprocess data
    df = pd.read_csv(config['data_path'])
    df_class_0_train, df_class_0_val, df_class_0_test, df_class_1 = preprocess(df, take_sample=True, sample=config['sample_size'])

    # Create datasets and dataloaders
    train_dataset = CustomDataset(df_class_0_train)
    val_dataset = CustomDataset(df_class_0_val)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    vae_model = VariationalAutoEncoder(input_dim=config['input_dim'], 
                                       hidden_dim=config['hidden_dim'], 
                                       z_dim=config['z_dim'])

    # Initialize optimizer
    optimizer = optim.Adam(vae_model.parameters(), lr=config['learning_rate'])

    # Training loop
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(config['epochs']), desc="Training epochs"):
        vae_model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training", leave=False):
            optimizer.zero_grad()
            reconstructed_batch, mu, logvar = vae_model(batch)
            loss = vae_loss_function(reconstructed_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation step
        vae_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} validation", leave=False):
                reconstructed_batch, mu, logvar = vae_model(batch)
                loss = vae_loss_function(reconstructed_batch, batch, mu, logvar)
                val_loss += loss.item()

        # Compute average losses
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Print losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{config['epochs']}], "
                       f"Training Loss: {avg_train_loss:.4f}, "
                       f"Validation Loss: {avg_val_loss:.4f}")

    # Create directory for saving model if it doesn't exist
    save_dir = os.path.join('models', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(save_dir, 'vae_model.pth')
    torch.save(vae_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    return vae_model, df_class_0_test, df_class_1, save_dir

def evaluate_model(model, df_class_0_test, df_class_1, batch_size, save_dir):
    # Create datasets and dataloaders
    test_dataset_class_0 = CustomDataset(df_class_0_test)
    test_dataset_class_1 = CustomDataset(df_class_1)
    
    test_loader_class_0 = DataLoader(test_dataset_class_0, batch_size=batch_size, shuffle=False)
    test_loader_class_1 = DataLoader(test_dataset_class_1, batch_size=batch_size, shuffle=False)

    # Calculate average reconstruction error for class 0 (normal transactions)
    avg_error_class_0 = calculate_reconstruction_error(model, test_loader_class_0)
    print(f"Average reconstruction error for normal transactions (Class 0): {avg_error_class_0:.4f}")

    # Calculate average reconstruction error for class 1 (fraudulent transactions)
    avg_error_class_1 = calculate_reconstruction_error(model, test_loader_class_1)
    print(f"Average reconstruction error for fraudulent transactions (Class 1): {avg_error_class_1:.4f}")

    # Calculate reconstruction errors for individual samples
    model.eval()
    errors_class_0 = []
    errors_class_1 = []

    with torch.no_grad():
        for batch in tqdm(test_loader_class_0, desc="Processing Class 0"):
            reconstructed_batch, _, _ = model(batch)
            error = torch.mean((reconstructed_batch - batch) ** 2, dim=1)
            errors_class_0.extend(error.tolist())

        for batch in tqdm(test_loader_class_1, desc="Processing Class 1"):
            reconstructed_batch, _, _ = model(batch)
            error = torch.mean((reconstructed_batch - batch) ** 2, dim=1)
            errors_class_1.extend(error.tolist())

    # Sample from class 0 to match the size of class 1
    sampled_errors_class_0 = pd.Series(errors_class_0).sample(n=len(errors_class_1), random_state=42)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(sampled_errors_class_0, bins=50, alpha=0.5, label='Normal Transactions (Class 0)')
    plt.hist(errors_class_1, bins=50, alpha=0.5, label='Fraudulent Transactions (Class 1)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'reconstruction_error_histogram.png'))
    plt.close()

if __name__ == "__main__":
    config = load_config('config.yaml')
    trained_model, df_class_0_test, df_class_1, save_dir = train_vae(config)
    print("Training completed. Model saved and loss plot generated.")
    
    evaluate_model(trained_model, df_class_0_test, df_class_1, config['batch_size'], save_dir)
    print("Evaluation completed. Reconstruction error histogram generated.")