# Enhanced Training Visualizations

Based on your feedback, I'll add more comprehensive visualizations to monitor the training progress. Here are the additional visualization components that will be incorporated into the notebook:

## 1. Real-time Training Progress Dashboard

```python
def create_training_dashboard():
    """Create a real-time training progress dashboard"""
    
    # Initialize dashboard data
    dashboard_data = {
        'epochs': [],
        'train_loss': [],
        'val_accuracy': [],
        'learning_rate': [],
        'time_elapsed': [],
        'memory_usage': []
    }
    
    # Create initial plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fire Detection AI - Training Progress Dashboard', fontsize=16)
    
    # Loss plot
    loss_line, = axes[0, 0].plot([], [], 'b-', label='Training Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Accuracy plot
    acc_line, = axes[0, 1].plot([], [], 'g-', label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    
    # Learning rate plot
    lr_line, = axes[1, 0].plot([], [], 'r-', label='Learning Rate')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True)
    
    # Memory usage plot
    mem_line, = axes[1, 1].plot([], [], 'm-', label='GPU Memory Usage (GB)')
    axes[1, 1].set_title('GPU Memory Usage')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Memory (GB)')
    axes[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, axes, dashboard_data

def update_dashboard(fig, axes, dashboard_data, epoch, loss, accuracy, lr, time_elapsed, memory_usage):
    """Update the training dashboard with new data"""
    
    # Update data
    dashboard_data['epochs'].append(epoch)
    dashboard_data['train_loss'].append(loss)
    dashboard_data['val_accuracy'].append(accuracy)
    dashboard_data['learning_rate'].append(lr)
    dashboard_data['time_elapsed'].append(time_elapsed)
    dashboard_data['memory_usage'].append(memory_usage)
    
    # Update plots
    axes[0, 0].plot(dashboard_data['epochs'], dashboard_data['train_loss'], 'b-')
    axes[0, 1].plot(dashboard_data['epochs'], dashboard_data['val_accuracy'], 'g-')
    axes[1, 0].plot(dashboard_data['epochs'], dashboard_data['learning_rate'], 'r-')
    axes[1, 1].plot(dashboard_data['epochs'], dashboard_data['memory_usage'], 'm-')
    
    # Update limits
    for i in range(2):
        for j in range(2):
            axes[i, j].relim()
            axes[i, j].autoscale_view()
    
    # Add current values as text
    plt.figtext(0.5, 0.01, f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.4f} | Time: {time_elapsed:.1f}s", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    # Refresh the figure
    fig.canvas.draw()
    display.clear_output(wait=True)
    display.display(fig)
```

## 2. Class Distribution Visualization

```python
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
    plt.savefig('class_distribution.png')
    plt.show()
```

## 3. Area-specific Performance Analysis

```python
def visualize_area_performance(transformer_model, ml_models, X_test, y_test, areas_test, area_names, device):
    """Visualize model performance by area"""
    
    # Get predictions from transformer
    transformer_model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    areas_test_tensor = torch.LongTensor(areas_test).to(device)
    
    with torch.no_grad():
        transformer_outputs = transformer_model(X_test_tensor, areas_test_tensor)
        transformer_preds = torch.argmax(transformer_outputs['fire_logits'], dim=1).cpu().numpy()
    
    # Get predictions from ML models
    X_test_features = engineer_features(X_test)
    X_test_scaled = scaler.transform(X_test_features)
    
    ml_preds = {}
    for name, model in ml_models.items():
        ml_preds[name] = model.predict(X_test_scaled)
    
    # Calculate accuracy by area
    area_accuracies = {}
    for area_idx, area_name in enumerate(area_names):
        # Filter test data by area
        area_mask = areas_test == area_idx
        area_y_test = y_test[area_mask]
        
        if len(area_y_test) == 0:
            continue
        
        # Calculate transformer accuracy
        area_transformer_preds = transformer_preds[area_mask]
        transformer_acc = np.mean(area_transformer_preds == area_y_test)
        
        # Calculate ML model accuracies
        ml_accs = {}
        for name, preds in ml_preds.items():
            area_ml_preds = preds[area_mask]
            ml_accs[name] = np.mean(area_ml_preds == area_y_test)
        
        area_accuracies[area_name] = {
            'transformer': transformer_acc,
            **ml_accs
        }
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(area_names))
    width = 0.15
    n_models = len(ml_models) + 1
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    
    # Plot transformer accuracy
    transformer_accs = [area_accuracies[area]['transformer'] for area in area_names]
    ax.bar(x + offsets[0], transformer_accs, width, label='Transformer')
    
    # Plot ML model accuracies
    for i, name in enumerate(ml_models.keys()):
        model_accs = [area_accuracies[area][name] for area in area_names]
        ax.bar(x + offsets[i+1], model_accs, width, label=name)
    
    ax.set_title('Model Performance by Area')
    ax.set_xticks(x)
    ax.set_xticklabels(area_names)
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.8, 1.0])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('area_performance.png')
    plt.show()
```

## 4. Confusion Matrix Visualization

```python
def plot_confusion_matrices(y_test, ensemble_preds, transformer_preds):
    """Plot confusion matrices for ensemble and transformer predictions"""
    
    class_names = ['Normal', 'Warning', 'Fire']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Ensemble confusion matrix
    cm_ensemble = confusion_matrix(y_test, ensemble_preds)
    cm_ensemble_norm = cm_ensemble.astype('float') / cm_ensemble.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_ensemble_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax1)
    ax1.set_title('Ensemble Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    
    # Add raw counts as text
    for i in range(cm_ensemble.shape[0]):
        for j in range(cm_ensemble.shape[1]):
            ax1.text(j+0.5, i+0.8, f"({cm_ensemble[i, j]})", 
                    ha="center", va="center", color="black", fontsize=9)
    
    # Transformer confusion matrix
    cm_transformer = confusion_matrix(y_test, transformer_preds)
    cm_transformer_norm = cm_transformer.astype('float') / cm_transformer.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_transformer_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2)
    ax2.set_title('Transformer Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_xticklabels(class_names)
    ax2.set_yticklabels(class_names)
    
    # Add raw counts as text
    for i in range(cm_transformer.shape[0]):
        for j in range(cm_transformer.shape[1]):
            ax2.text(j+0.5, i+0.8, f"({cm_transformer[i, j]})", 
                    ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()
```

## 5. Training Progress Animation

```python
def create_training_animation(training_history):
    """Create an animated GIF of training progress"""
    
    # Extract metrics from training history
    epochs = [entry['epoch'] for entry in training_history]
    losses = [entry['loss'] for entry in training_history]
    accuracies = [entry['val_accuracy'] for entry in training_history]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training Progress Animation', fontsize=16)
    
    # Initialize plots
    loss_line, = ax1.plot([], [], 'b-', label='Training Loss')
    acc_line, = ax2.plot([], [], 'g-', label='Validation Accuracy')
    
    ax1.set_xlim(0, len(epochs))
    ax1.set_ylim(0, max(losses) * 1.1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlim(0, len(epochs))
    ax2.set_ylim(min(accuracies) * 0.9, 1.0)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Animation function
    def animate(i):
        loss_line.set_data(range(i+1), losses[:i+1])
        acc_line.set_data(range(i+1), accuracies[:i+1])
        
        # Add current values as text
        plt.figtext(0.5, 0.01, f"Epoch: {i+1} | Loss: {losses[i]:.4f} | Accuracy: {accuracies[i]:.4f}", 
                    ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        return loss_line, acc_line
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(epochs), interval=500, blit=True)
    
    # Save as GIF
    anim.save('training_progress.gif', writer='pillow', fps=2)
    
    plt.close()
    
    # Display the GIF
    display.HTML('<img src="training_progress.gif">')
```

## 6. Feature Importance Visualization

```python
def visualize_feature_importance(ml_models, feature_names):
    """Visualize feature importance from ML models"""
    
    # Get feature importance from models
    importances = {}
    
    if 'random_forest' in ml_models:
        importances['Random Forest'] = ml_models['random_forest'].feature_importances_
    
    if 'xgboost' in ml_models and hasattr(ml_models['xgboost'], 'feature_importances_'):
        importances['XGBoost'] = ml_models['xgboost'].feature_importances_
    
    if 'lightgbm' in ml_models and hasattr(ml_models['lightgbm'], 'feature_importances_'):
        importances['LightGBM'] = ml_models['lightgbm'].feature_importances_
    
    if not importances:
        print("No feature importance available from models")
        return
    
    # Create figure
    n_models = len(importances)
    fig, axes = plt.subplots(n_models, 1, figsize=(10, 5*n_models))
    
    if n_models == 1:
        axes = [axes]
    
    # Plot feature importance for each model
    for i, (name, importance) in enumerate(importances.items()):
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        top_n = min(20, len(indices))  # Show top 20 features
        
        # Plot
        axes[i].barh(range(top_n), importance[indices[:top_n]])
        axes[i].set_yticks(range(top_n))
        axes[i].set_yticklabels([feature_names[j] for j in indices[:top_n]])
        axes[i].set_title(f'{name} Feature Importance')
        axes[i].set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
```

## 7. Learning Curves Visualization

```python
def plot_learning_curves(training_history):
    """Plot learning curves from training history"""
    
    # Extract metrics
    epochs = [entry['epoch'] for entry in training_history]
    train_losses = [entry['train_loss'] for entry in training_history]
    val_accuracies = [entry['val_accuracy'] for entry in training_history]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', marker='o', label='Training Loss')
    ax1.set_title('Training Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Add moving average
    window_size = min(5, len(train_losses))
    if window_size > 1:
        moving_avg = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(epochs[window_size-1:], moving_avg, 'r--', label=f'{window_size}-epoch Moving Avg')
        ax1.legend()
    
    # Accuracy curve
    ax2.plot(epochs, val_accuracies, 'g-', marker='o', label='Validation Accuracy')
    ax2.set_title('Validation Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # Add moving average
    if window_size > 1:
        moving_avg = np.convolve(val_accuracies, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(epochs[window_size-1:], moving_avg, 'r--', label=f'{window_size}-epoch Moving Avg')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()
```

## 8. Training Time Breakdown

```python
def visualize_training_time_breakdown(time_metrics):
    """Visualize breakdown of training time by component"""
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    labels = list(time_metrics.keys())
    sizes = list(time_metrics.values())
    total_time = sum(sizes)
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title('Training Time Breakdown')
    
    # Bar chart
    ax2.barh(labels, sizes)
    ax2.set_title('Training Time by Component')
    ax2.set_xlabel('Time (seconds)')
    
    # Add time labels
    for i, v in enumerate(sizes):
        ax2.text(v + 0.1, i, f"{v:.1f}s ({v/total_time:.1%})", va='center')
    
    plt.tight_layout()
    plt.savefig('time_breakdown.png')
    plt.show()
```

## 9. Memory Usage Monitoring

```python
def monitor_memory_usage():
    """Monitor and visualize memory usage during training"""
    
    # Initialize memory tracking
    memory_usage = []
    timestamps = []
    
    if torch.cuda.is_available():
        # Get initial memory usage
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1e9  # GB
        
        memory_usage.append(initial_memory)
        timestamps.append(0)
        
        def track_memory():
            current_memory = torch.cuda.memory_allocated() / 1e9  # GB
            peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
            memory_usage.append(current_memory)
            timestamps.append(time.time() - start_time)
            
            return current_memory, peak_memory
    else:
        # CPU memory tracking (psutil)
        try:
            import psutil
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1e9  # GB
            
            memory_usage.append(initial_memory)
            timestamps.append(0)
            
            def track_memory():
                current_memory = process.memory_info().rss / 1e9  # GB
                memory_usage.append(current_memory)
                timestamps.append(time.time() - start_time)
                
                return current_memory, max(memory_usage)
        except:
            # Fallback if psutil not available
            def track_memory():
                return 0, 0
    
    return track_memory, memory_usage, timestamps
```

## 10. Interactive Model Comparison

```python
def create_interactive_model_comparison(ml_results, transformer_acc, ensemble_acc):
    """Create interactive model comparison visualization"""
    
    # Combine all results
    all_results = {
        'Transformer': transformer_acc,
        'Ensemble': ensemble_acc,
        **ml_results
    }
    
    # Sort by accuracy
    sorted_models = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    models = [item[0] for item in sorted_models]
    accuracies = [item[1] for item in sorted_models]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = ax.barh(models, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(models))))
    
    # Add accuracy values
    for i, v in enumerate(accuracies):
        ax.text(v + 0.005, i, f"{v:.4f} ({v*100:.1f}%)", va='center')
    
    # Add target line
    ax.axvline(x=0.95, color='r', linestyle='--', label='Target (95%)')
    ax.text(0.95, len(models) + 0.2, 'Target: 95%', color='r')
    
    # Customize plot
    ax.set_title('Model Accuracy Comparison', fontsize=16)
    ax.set_xlabel('Accuracy')
    ax.set_xlim([0.85, 1.0])
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Highlight best model
    best_model = models[0]
    best_acc = accuracies[0]
    ax.get_children()[0].set_color('gold')
    ax.text(0.86, 0, f"Best: {best_model} ({best_acc:.4f})", 
            fontsize=12, bbox=dict(facecolor='gold', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
```

## Integration into Training Process

To integrate these visualizations into the training process, we'll update the main training function to include these visualizations at appropriate points:

```python
def main():
    """Main training function with enhanced visualizations"""
    
    logger.info("🔥" * 80)
    logger.info("FIRE DETECTION AI - 5M DATASET OPTIMIZED TRAINING")
    logger.info("🔥" * 80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    start_time = time.time()
    
    # Initialize memory monitoring
    memory_tracker, memory_usage, timestamps = monitor_memory_usage()
    
    # Load data
    data_loader = OptimizedDataLoader()
    X, y, areas = data_loader.load_all_data_sample()
    
    # Split data
    X_train, X_test, y_train, y_test, areas_train, areas_test = train_test_split(
        X, y, areas, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train, X_val, y_train, y_val, areas_train, areas_val = train_test_split(
        X_train, y_train, areas_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )
    
    logger.info(f"📊 Data splits:")
    logger.info(f"   Training: {len(X_train):,}")
    logger.info(f"   Validation: {len(X_val):,}")
    logger.info(f"   Test: {len(X_test):,}")
    
    # Visualize class distribution
    visualize_class_distribution(y_train, y_val, y_test)
    
    # Initialize training dashboard
    fig, axes, dashboard_data = create_training_dashboard()
    
    # Initialize training history
    training_history = []
    
    # Initialize time metrics
    time_metrics = {
        'Data Loading': time.time() - start_time,
        'Transformer Training': 0,
        'ML Ensemble Training': 0,
        'Evaluation': 0,
        'Visualization': 0
    }
    
    # Train transformer with early stopping and visualization
    transformer_start = time.time()
    transformer_model, transformer_acc, transformer_history = train_transformer_with_early_stopping(
        X_train, y_train, areas_train, X_val, y_val, areas_val, device,
        dashboard_fig=fig, dashboard_axes=axes, dashboard_data=dashboard_data,
        memory_tracker=memory_tracker
    )
    time_metrics['Transformer Training'] = time.time() - transformer_start
    
    # Visualize learning curves
    plot_learning_curves(transformer_history)
    
    # Train ML ensemble
    ml_start = time.time()
    ml_models, ml_results, scaler = train_optimized_ml_ensemble(X_train, y_train, X_val, y_val)
    time_metrics['ML Ensemble Training'] = time.time() - ml_start
    
    # Visualize feature importance
    feature_names = [f"Feature_{i}" for i in range(X_train_features.shape[1])]
    visualize_feature_importance(ml_models, feature_names)
    
    # Final evaluation
    eval_start = time.time()
    ensemble_acc, transformer_preds, ensemble_preds = evaluate_ensemble(
        transformer_model, ml_models, scaler, X_test, y_test, areas_test, device
    )
    time_metrics['Evaluation'] = time.time() - eval_start
    
    # Visualize confusion matrices
    plot_confusion_matrices(y_test, ensemble_preds, transformer_preds)
    
    # Visualize area performance
    area_names = list(data_loader.area_files.keys())
    visualize_area_performance(transformer_model, ml_models, X_test, y_test, areas_test, area_names, device)
    
    # Create interactive model comparison
    create_interactive_model_comparison(ml_results, transformer_acc, ensemble_acc)
    
    # Visualize training time breakdown
    visualize_training_time_breakdown(time_metrics)
    
    # Create training progress animation
    create_training_animation(transformer_history)
    
    # Save models
    save_models_to_s3(transformer_model, ml_models, scaler, ensemble_acc)
    
    # Visualize performance comparison with 50M model
    vis_start = time.time()
    total_time = time.time() - start_time
    memory_usage = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    visualize_performance(ensemble_acc, total_time, memory_usage)
    time_metrics['Visualization'] = time.time() - vis_start
    
    # Final summary
    logger.info("\n" + "🎉" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("🎉" * 80)
    logger.info(f"🏆 Final Ensemble Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.1f}%)")
    logger.info(f"⏱️ Total training time: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"📊 Total samples processed: {len(X