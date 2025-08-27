class OptimizedDataLoader:
    """Simplified data loader for sequence creation from existing data"""
    
    def __init__(self):
        """Initialize the data loader"""
        pass
    
    def create_sequences(self, X, y, seq_len=60, step=10):
        """Create time series sequences from existing data
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target labels array of shape (n_samples,)
            seq_len: Length of each sequence
            step: Step size between sequences
            
        Returns:
            sequences: Array of shape (n_sequences, seq_len, n_features)
            labels: Array of shape (n_sequences,)
        """
        
        logger.info(f"🔄 Creating sequences with length {seq_len} and step {step}")
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - seq_len, step):
            sequences.append(X[i:i+seq_len])
            labels.append(y[i+seq_len-1])  # Use label of last timestep
        
        logger.info(f"   ✅ Created {len(sequences):,} sequences from {len(X):,} samples")
        
        return np.array(sequences), np.array(labels)