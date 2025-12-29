# Fire Detection AI - 5M Dataset Training (Combined Notebook)
# PART B: Model Architecture and Training

# This is part B of the combined notebook. Copy and paste all parts in sequence into your SageMaker notebook.

# ===== CELL 10: Create PyTorch Datasets and DataLoaders =====
class FireDetectionDataset(Dataset):
    """PyTorch Dataset for Fire Detection data"""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
train_dataset = FireDetectionDataset(X_train, y_train)
val_dataset = FireDetectionDataset(X_val, y_val)
test_dataset = FireDetectionDataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)} batches")

# ===== CELL 11: Define Model Architecture =====
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class FireDetectionTransformer(nn.Module):
    """Transformer model for fire detection"""
    
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_classes, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        
        # Project to output classes
        x = self.output_projection(x)
        
        return x

# Create model
input_dim = X_train.shape[2]  # Number of features
num_classes = len(np.unique(y_train))

model = FireDetectionTransformer(
    input_dim=input_dim,
    d_model=TRANSFORMER_CONFIG['d_model'],
    num_heads=TRANSFORMER_CONFIG['num_heads'],
    num_layers=TRANSFORMER_CONFIG['num_layers'],
    num_classes=num_classes,
    dropout=TRANSFORMER_CONFIG['dropout']
)

# Move model to device
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs for data parallel training")
    model = nn.DataParallel(model)

model = model.to(device)

# Print model summary
logger.info(f"Model architecture:")
logger.info(f"{model}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Total parameters: {total_params:,}")
logger.info(f"Trainable parameters: {trainable_params:,}")

# ===== CELL 12: Define Loss Function and Optimizer =====
# Define class weights to handle imbalance
class_counts = np.bincount(y_train.astype(int))
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

logger.info(f"Class weights: {class_weights.cpu().numpy()}")

# Define loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=3, 
    verbose=True
)

# ===== CELL 13: Define Training and Evaluation Functions =====
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in pbar:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / total,
            'acc': 100. * correct / total
        })
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Store targets and predictions for metrics
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    
    # Calculate epoch statistics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Combine all targets and predictions
    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    
    # Calculate additional metrics
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_targets, all_predictions

# ===== CELL 14: Training Loop with Early Stopping =====
@error_handler
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, patience, device, checkpoint_dir='checkpoints'):
    """Train model with early stopping"""
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = None
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'lr': []
    }
    
    # Create visualization objects
    if VISUALIZATION_CONFIG['update_interval'] > 0:
        # Create figure for loss and accuracy
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        train_loss_line, = plt.plot([], [], 'b-', label='Train Loss')
        val_loss_line, = plt.plot([], [], 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        train_acc_line, = plt.plot([], [], 'b-', label='Train Acc')
        val_acc_line, = plt.plot([], [], 'r-', label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    # Training loop
    start_time = time.time()
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on validation set
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        logger.info(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s - "
                   f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                   f"Val F1: {val_f1:.4f} - LR: {current_lr:.6f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'history': history
            }, checkpoint_path)
            
            best_model_path = checkpoint_path
            logger.info(f"✅ Saved new best model to {checkpoint_path}")
        
        # Update visualization
        if VISUALIZATION_CONFIG['update_interval'] > 0 and (epoch + 1) % VISUALIZATION_CONFIG['update_interval'] == 0:
            epochs = list(range(1, epoch + 2))
            
            train_loss_line.set_data(epochs, history['train_loss'])
            val_loss_line.set_data(epochs, history['val_loss'])
            train_acc_line.set_data(epochs, history['train_acc'])
            val_acc_line.set_data(epochs, history['val_acc'])
            
            plt.subplot(1, 2, 1)
            plt.xlim(1, num_epochs)
            plt.ylim(0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1)
            
            plt.subplot(1, 2, 2)
            plt.xlim(1, num_epochs)
            plt.ylim(0, 1)
            
            plt.draw()
            plt.pause(0.1)
            
            # Save figure if enabled
            if VISUALIZATION_CONFIG['save_figures']:
                plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/training_progress_epoch_{epoch+1}.png", dpi=300)
            
            display.clear_output(wait=True)
            display.display(plt.gcf())
        
        # Check for early stopping
        if epoch - best_epoch >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            print_gpu_memory()
    
    # Calculate total training time
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
    logger.info(f"Best model at epoch {best_epoch+1} with validation loss {best_val_loss:.4f}")
    
    # Load best model
    if best_model_path is not None:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from {best_model_path}")
    
    return model, history

# ===== CELL 15: Train Model =====
# Train model
model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=EPOCHS,
    patience=EARLY_STOPPING_PATIENCE,
    device=device
)

# ===== CELL 16: Visualize Training History =====
def plot_training_history(history):
    """Plot training history"""
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Plot F1 score
    ax3.plot(epochs, history['val_precision'], 'g-', label='Precision')
    ax3.plot(epochs, history['val_recall'], 'b-', label='Recall')
    ax3.plot(epochs, history['val_f1'], 'r-', label='F1 Score')
    ax3.set_title('Validation Metrics')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True)
    
    # Plot learning rate
    ax4.plot(epochs, history['lr'], 'k-')
    ax4.set_title('Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/training_history.png", dpi=300)
    
    plt.show()

# Plot training history
plot_training_history(history)