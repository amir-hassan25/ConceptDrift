import os
import torch


def save_model(path, epoch, memory, gnn, link_pred, optimizer, loss, model_dir="saved_models"):
    # Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define the save path for the model
    model_path = path
    
    # Save the state dictionaries of the memory, gnn, link predictor, and optimizer
    torch.save({
        'epoch': epoch,
        'memory_state_dict': memory.state_dict(),
        'gnn_state_dict': gnn.state_dict(),
        'link_pred_state_dict': link_pred.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, model_path)
    
    print(f"Model saved to {model_path}")


def load_model(model_path, memory, gnn, link_pred, optimizer=None):
    # Load the saved model checkpoint
    checkpoint = torch.load(model_path)

    # Restore the state dictionaries of the components
    memory.load_state_dict(checkpoint['memory_state_dict'])
    gnn.load_state_dict(checkpoint['gnn_state_dict'])
    link_pred.load_state_dict(checkpoint['link_pred_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the epoch and loss from the checkpoint
    return checkpoint['epoch'], checkpoint['loss']