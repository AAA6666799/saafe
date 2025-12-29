def prepare_dataloaders(X, y, batch_size=32, val_split=0.2, test_split=0.1, seed=42):
    """
    Prepare train, validation, and test dataloaders with balanced sampling.
    
    Args:
        X: Feature tensor
        y: Label tensor
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get dataset size
    num_samples = X.shape[0]
    
    # Create indices for splits
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    test_size = int(num_samples * test_split)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size - test_size
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])
    
    # Create balanced sampler for training set
    # Count samples per class
    y_train = y[train_indices]
    class_counts = torch.bincount(y_train.view(-1))
    
    # Calculate weights
    class_weights = 1.0 / class_counts.float()
    sample_weights = torch.zeros(len(train_dataset))
    
    for i in range(len(train_dataset)):
        # Get the most common class in this sequence
        class_idx = torch.mode(train_dataset[i][1]).values.item()
        sample_weights[i] = class_weights[class_idx]
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Created dataloaders with {len(train_dataset)} training, "
               f"{len(val_dataset)} validation, and {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader


def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot validation F1 score
    plt.subplot(2, 2, 3)
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()


def plot_class_transitions(y_true, y_pred, class_names):
    """Plot class transitions for true and predicted sequences"""
    # Sample a few sequences
    num_samples = min(5, len(y_true))
    sample_indices = np.random.choice(len(y_true), num_samples, replace=False)
    
    plt.figure(figsize=(15, 10))
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(y_true[idx], label='True', marker='o', markersize=4)
        plt.plot(y_pred[idx], label='Predicted', marker='x', markersize=4)
        plt.yticks(range(len(class_names)), class_names)
        plt.title(f'Sequence {idx}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_transitions.png'))
    plt.close()


def setup_model(device):
    """Setup model, loss function, optimizer, and scheduler"""
    # Create model
    model = EnhancedFireDetectionTransformer(
        input_dim=5,
        hidden_dim=128,
        num_classes=3,
        num_layers=4,
        num_heads=4,
        dropout=0.2,
        use_cnn_branch=True,
        use_early_warning=True
    ).to(device)
    
    # Create loss function
    criterion = WarningFocusedLoss(
        warning_weight=3.0,
        normal_to_warning_weight=2.0,
        warning_to_fire_weight=2.0
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=LEARNING_RATE / 10
    )
    
    return model, criterion, optimizer, lr_scheduler


def save_model_summary(model):
    """Save model summary to file"""
    # Capture model summary
    summary_file = os.path.join(OUTPUT_DIR, 'model_summary.txt')
    with open(summary_file, 'w') as f:
        # Redirect stdout to file
        with redirect_stdout(f):
            # Create dummy input
            dummy_input = torch.zeros(1, 100, 5).to(next(model.parameters()).device)
            # Print model
            print(model)
            # Print parameter count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            # Try to print shapes
            try:
                summary(model, input_data=dummy_input)
            except:
                print("Could not generate detailed summary")
    
    logger.info(f"Saved model summary to {summary_file}")


def main():
    """Main execution function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Print configuration
    logger.info("Configuration:")
    logger.info(f"- Random seed: {RANDOM_SEED}")
    logger.info(f"- Batch size: {BATCH_SIZE}")
    logger.info(f"- Number of epochs: {NUM_EPOCHS}")
    logger.info(f"- Learning rate: {LEARNING_RATE}")
    logger.info(f"- Weight decay: {WEIGHT_DECAY}")
    logger.info(f"- Gradient clip value: {GRADIENT_CLIP_VALUE}")
    logger.info(f"- Sequence length: {SEQUENCE_LENGTH}")
    logger.info(f"- Number of samples: {NUM_SAMPLES}")
    
    # Train enhanced model
    model, test_acc, test_f1 = train_enhanced_model(device)
    
    # Print final results
    logger.info("Final results:")
    logger.info(f"- Test accuracy: {test_acc:.4f}")
    logger.info(f"- Test F1 score: {test_f1:.4f}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()