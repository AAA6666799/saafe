# This code assumes you already have a synthetic dataset loaded
# If you don't have X, y, and areas variables defined, you need to load them first

# Initialize data loader for sequence creation and other utilities
data_loader = OptimizedDataLoader()

# If you already have X, y, and areas loaded, you can create sequences directly
# Make sure X, y, and areas are already defined before running this code

# Create sequences from the existing data
X_sequences = []
y_sequences = []
areas_sequences = []

# Process each area separately
unique_areas = np.unique(areas)
for area_idx in unique_areas:
    area_mask = areas == area_idx
    X_area = X[area_mask]
    y_area = y[area_mask]
    
    if len(X_area) > 60:  # Minimum sequence length
        # Create sequences for this area
        logger.info(f"Creating sequences for area {area_idx}")
        X_seq, y_seq = data_loader.create_sequences(X_area, y_area, seq_len=60, step=10)
        areas_seq = np.full(len(X_seq), area_idx)
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        areas_sequences.append(areas_seq)
        logger.info(f"Created {len(X_seq)} sequences for area {area_idx}")

# Combine sequences from all areas
if X_sequences:
    X = np.vstack(X_sequences)
    y = np.hstack(y_sequences)
    areas = np.hstack(areas_sequences)
    logger.info(f"Combined sequences: X={X.shape}, y={y.shape}, areas={areas.shape}")
else:
    logger.error("No sequences created. Check your input data.")

# Split data into train/validation/test sets
X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
    X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
    X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
)

# Print data split summary
logger.info(f"📊 Data splits:")
logger.info(f"   Train: {X_train.shape}, {np.bincount(y_train.astype(int))}")
logger.info(f"   Validation: {X_val.shape}, {np.bincount(y_val.astype(int))}")
logger.info(f"   Test: {X_test.shape}, {np.bincount(y_test.astype(int))}")

# Visualize class distribution
def visualize_class_distribution(y_train, y_val, y_test):
    """Visualize class distribution across train/val/test sets"""
    
    class_names = ['Normal', 'Warning', 'Fire']
    
    # Count classes in each set
    train_counts = np.bincount(y_train.astype(int), minlength=3)
    val_counts = np.bincount(y_val.astype(int), minlength=3)
    test_counts = np.bincount(y_test.astype(int), minlength=3)
    
    # Convert to percentages
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw counts
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train')
    ax1.bar(x, val_counts, width, label='Validation')
    ax1.bar(x + width, test_counts, width, label='Test')
    
    ax1.set_title('Class Distribution (Counts)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.set_ylabel('Number of Samples')
    ax1.legend()
    
    # Add count labels
    for i, v in enumerate(train_counts):
        ax1.text(i - width, v + 0.1, f"{v:,}", ha='center')
    for i, v in enumerate(val_counts):
        ax1.text(i, v + 0.1, f"{v:,}", ha='center')
    for i, v in enumerate(test_counts):
        ax1.text(i + width, v + 0.1, f"{v:,}", ha='center')
    
    # Percentages
    ax2.bar(x - width, train_pct, width, label='Train')
    ax2.bar(x, val_pct, width, label='Validation')
    ax2.bar(x + width, test_pct, width, label='Test')
    
    ax2.set_title('Class Distribution (Percentage)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.set_ylabel('Percentage (%)')
    ax2.legend()
    
    # Add percentage labels
    for i, v in enumerate(train_pct):
        ax2.text(i - width, v + 0.5, f"{v:.1f}%", ha='center')
    for i, v in enumerate(val_pct):
        ax2.text(i, v + 0.5, f"{v:.1f}%", ha='center')
    for i, v in enumerate(test_pct):
        ax2.text(i + width, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/class_distribution.png", dpi=300)
    
    plt.show()

# Visualize the class distribution
visualize_class_distribution(y_train, y_val, y_test)