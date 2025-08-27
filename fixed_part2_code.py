# Data Loading and Sampling

## Optimized Data Loader
class OptimizedDataLoader:
    """Optimized data loader with sampling for 5M dataset"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3') if AWS_AVAILABLE else None
        self.area_files = {
            'basement': 'basement_data_cleaned.csv',
            'laundry': 'laundry_data_cleaned.csv',
            'asd': 'asd_data_cleaned.csv',
            'voc': 'voc_data_cleaned.csv',
            'arc': 'arc_data_cleaned.csv'
        }
        self.area_to_idx = {area: idx for idx, area in enumerate(self.area_files.keys())}
    
    @error_handler
    def analyze_dataset_structure(self):
        """Analyze the structure of the full dataset"""
        if not self.s3_client:
            logger.error("❌ S3 client not available")
            return {}
        
        # Get list of all dataset files
        response = self.s3_client.list_objects_v2(Bucket=DATASET_BUCKET, Prefix=DATASET_PREFIX)
        
        dataset_stats = {}
        total_size = 0
        
        for obj in response.get('Contents', []):
            key = obj['Key']
            size = obj['Size']
            total_size += size
            
            # Extract area name from key
            area_name = key.split('/')[-1].split('_')[0]
            
            if area_name not in dataset_stats:
                dataset_stats[area_name] = {
                    'files': [],
                    'total_size': 0,
                    'estimated_rows': 0
                }
            
            dataset_stats[area_name]['files'].append(key)
            dataset_stats[area_name]['total_size'] += size
            # Rough estimate: assume each sample is ~200 bytes on average
            dataset_stats[area_name]['estimated_rows'] += size // 200
        
        # Print summary
        logger.info(f"📊 Dataset Analysis Summary:")
        logger.info(f"   Total size: {total_size / (1024*1024*1024):.2f} GB")
        
        for area, stats in dataset_stats.items():
            logger.info(f"   {area}: {stats['estimated_rows']:,} rows, {stats['total_size'] / (1024*1024*1024):.2f} GB")
        
        return dataset_stats
    
    @error_handler
    def load_area_data_sample(self, area_name, max_samples=SAMPLE_SIZE_PER_AREA):
        """Load and sample area data with error handling"""
        if not self.s3_client:
            logger.error("❌ S3 client not available")
            raise DataLoadingError("S3 client not available")
            
        file_key = f"{DATASET_PREFIX}{self.area_files[area_name]}"
        
        logger.info(f"📥 Loading {area_name}: s3://{DATASET_BUCKET}/{file_key}")
        
        try:
            # Check if file exists
            try:
                self.s3_client.head_object(Bucket=DATASET_BUCKET, Key=file_key)
            except Exception as e:
                raise DataLoadingError(f"File not found: s3://{DATASET_BUCKET}/{file_key}")
            
            # Load data in chunks with retry mechanism
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.s3_client.get_object(Bucket=DATASET_BUCKET, Key=file_key)
                    
                    # Use pandas to read CSV in chunks
                    chunk_size = 100000  # 100K rows per chunk
                    chunks = []
                    
                    # Stream data from S3 in chunks and sample
                    chunk_iter = pd.read_csv(response['Body'], chunksize=chunk_size)
                    
                    total_rows = 0
                    for i, chunk in enumerate(chunk_iter):
                        # Sample from chunk based on ratio
                        if max_samples and total_rows + len(chunk) > max_samples:
                            # Calculate how many more samples we need
                            samples_needed = max_samples - total_rows
                            if samples_needed > 0:
                                # Sample remaining rows
                                chunk = chunk.sample(n=samples_needed, random_state=RANDOM_SEED)
                            else:
                                break
                        
                        chunks.append(chunk)
                        total_rows += len(chunk)
                        
                        logger.info(f"   📊 Chunk {i+1}: {len(chunk):,} rows, Total: {total_rows:,}")
                        
                        # Stop if we have enough samples
                        if max_samples and total_rows >= max_samples:
                            break
                    
                    # Combine chunks
                    if not chunks:
                        raise DataLoadingError(f"No data loaded from {area_name}")
                    
                    df = pd.concat(chunks, ignore_index=True)
                    
                    logger.info(f"   📊 Loaded {len(df):,} rows from {area_name}")
                    
                    # Apply temporal pattern preservation if enabled
                    if PRESERVE_TEMPORAL_PATTERNS:
                        df = self._preserve_temporal_patterns(df)
                    
                    # Apply class balancing if enabled
                    if ENSURE_CLASS_BALANCE:
                        df = self._balance_classes(df)
                    
                    return self._preprocess_area_data(df, area_name)
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.warning(f"   ⚠️ Error loading {area_name}, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise DataLoadingError(f"Failed to load {area_name} after {max_retries} attempts: {e}")
            
        except DataLoadingError as e:
            # Re-raise specific errors
            raise
        except Exception as e:
            # Wrap other exceptions
            raise DataLoadingError(f"Error loading {area_name}: {e}")
    
    def _preserve_temporal_patterns(self, df, timestamp_col='timestamp'):
        """Ensure temporal patterns are preserved in the sampled data"""
        
        if timestamp_col in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Sort by timestamp
            df = df.sort_values(timestamp_col).reset_index(drop=True)
            
            # Ensure we have continuous segments
            # Calculate time differences
            time_diffs = df[timestamp_col].diff()
            
            # Find large gaps (e.g., > 1 hour)
            large_gaps = time_diffs > pd.Timedelta(hours=1)
            segment_starts = df.index[large_gaps].tolist()
            
            if segment_starts:
                logger.info(f"   ⚠️ Found {len(segment_starts)} temporal gaps in data")
                
                # Add start of dataframe as first segment
                segment_starts = [0] + segment_starts
                
                # Add end of dataframe as last segment
                segment_starts.append(len(df))
                
                # Create segments
                segments = []
                for i in range(len(segment_starts) - 1):
                    start = segment_starts[i]
                    end = segment_starts[i + 1]
                    segments.append(df.iloc[start:end])
                
                # Sample from each segment proportionally
                sampled_segments = []
                for segment in segments:
                    # Sample proportionally to segment size
                    sample_size = int(len(segment) / len(df) * len(df))
                    if sample_size > 0:
                        sampled_segment = segment.sample(n=min(sample_size, len(segment)), 
                                                        random_state=RANDOM_SEED)
                        sampled_segments.append(sampled_segment)
                
                # Combine segments and sort by timestamp
                df = pd.concat(sampled_segments).sort_values(timestamp_col).reset_index(drop=True)
        
        return df
    
    def _balance_classes(self, df):
        """Balance classes according to target distribution"""
        
        # Determine label column
        label_col = None
        if 'is_anomaly' in df.columns:
            label_col = 'is_anomaly'
        elif 'label' in df.columns:
            label_col = 'label'
        else:
            # Generate synthetic labels based on first feature
            feature_col = df.columns[0]
            values = df[feature_col].values
            q95 = np.percentile(values, 95)
            q85 = np.percentile(values, 85)
            
            df['synthetic_label'] = 0
            df.loc[values > q95, 'synthetic_label'] = 2  # Fire (top 5%)
            df.loc[(values > q85) & (values <= q95), 'synthetic_label'] = 1  # Warning (85-95%)
            label_col = 'synthetic_label'
        
        # Get current class counts
        class_counts = df[label_col].value_counts().to_dict()
        total_samples = len(df)
        
        # Calculate target counts
        target_counts = {}
        for cls, pct in CLASS_DISTRIBUTION_TARGET.items():
            target_counts[cls] = int(total_samples * pct)
        
        # Adjust to ensure we have exactly total_samples
        total_target = sum(target_counts.values())
        if total_target != total_samples:
            # Add/remove difference to/from majority class
            majority_class = max(class_counts, key=class_counts.get)
            target_counts[majority_class] += (total_samples - total_target)
        
        # Sample or duplicate to reach target counts
        balanced_samples = []
        
        for cls, target_count in target_counts.items():
            class_df = df[df[label_col] == cls]
            current_count = len(class_df)
            
            if current_count == 0:
                logger.warning(f"   ⚠️ No samples for class {cls}")
                continue
            
            if current_count > target_count:
                # Downsample
                sampled = class_df.sample(n=target_count, random_state=RANDOM_SEED)
            else:
                # Upsample with replacement
                sampled = class_df.sample(n=target_count, replace=True, random_state=RANDOM_SEED)
            
            balanced_samples.append(sampled)
        
        # Combine and shuffle
        balanced_df = pd.concat(balanced_samples).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        
        # Verify balanced distribution
        balanced_distribution = balanced_df[label_col].value_counts(normalize=True).to_dict()
        
        logger.info(f"   📊 Balanced class distribution:")
        logger.info(f"      Normal (0): {balanced_distribution.get(0, 0):.2%}")
        logger.info(f"      Warning (1): {balanced_distribution.get(1, 0):.2%}")
        logger.info(f"      Fire (2): {balanced_distribution.get(2, 0):.2%}")
        
        return balanced_df
    
    def _preprocess_area_data(self, df, area_name):
        """Preprocess area data for model training"""
        
        logger.info(f"🔧 Preprocessing {area_name}...")
        
        # Handle timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features
        exclude_cols = ['timestamp', 'is_anomaly', 'label', 'synthetic_label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Limit features by area type
        if area_name == 'basement':
            feature_cols = feature_cols[:4]
        elif area_name == 'laundry':
            feature_cols = feature_cols[:3]
        else:
            feature_cols = feature_cols[:2]
        
        X = df[feature_cols].fillna(0).values
        
        # Create intelligent labels
        if 'is_anomaly' in df.columns:
            y = df['is_anomaly'].values.astype(int)
        elif 'label' in df.columns:
            y = df['label'].values.astype(int)
        elif 'synthetic_label' in df.columns:
            y = df['synthetic_label'].values.astype(int)
        else:
            # Generate realistic fire detection labels
            values = df[feature_cols[0]].values
            q95 = np.percentile(values, 95)
            q85 = np.percentile(values, 85)
            
            y = np.zeros(len(values))
            y[values > q95] = 2  # Fire (top 5%)
            y[(values > q85) & (values <= q95)] = 1  # Warning (85-95%)
        
        # Standardize to 6 features
        if X.shape[1] < 6:
            padding = np.zeros((X.shape[0], 6 - X.shape[1]))
            X = np.hstack([X, padding])
        elif X.shape[1] > 6:
            X = X[:, :6]
        
        logger.info(f"   ✅ {area_name}: {X.shape}, anomaly_rate={np.mean(y > 0):.4f}")
        return X, y
    
    def create_sequences(self, X, y, seq_len=60, step=10):
        """Create time series sequences"""
        
        logger.info(f"🔄 Creating sequences with length {seq_len} and step {step}")
        
        sequences = []
        labels = []
        
        for i in range(0, len(X) - seq_len, step):
            sequences.append(X[i:i+seq_len])
            labels.append(y[i+seq_len-1])  # Use label of last timestep
        
        logger.info(f"   ✅ Created {len(sequences):,} sequences from {len(X):,} samples")
        
        return np.array(sequences), np.array(labels)
    
    @error_handler
    def load_all_data_sample(self):
        """Load and sample the complete dataset"""
        
        logger.info("🚀 LOADING AND SAMPLING DATASET FROM S3")
        logger.info("=" * 50)
        
        # Analyze dataset structure
        dataset_stats = self.analyze_dataset_structure()
        
        all_X = []
        all_y = []
        all_areas = []
        
        start_time = time.time()
        
        for area_idx, area_name in enumerate(self.area_files.keys()):
            logger.info(f"\n📁 PROCESSING AREA {area_idx+1}/5: {area_name.upper()}")
            
            X, y = self.load_area_data_sample(area_name, SAMPLE_SIZE_PER_AREA)
            if X is None:
                continue
            
            sequences, labels = self.create_sequences(X, y, seq_len=60, step=10)
            areas = np.full(len(sequences), area_idx)
            
            all_X.append(sequences)
            all_y.append(labels)
            all_areas.append(areas)
            
            logger.info(f"   ✅ Created {len(sequences):,} sequences")
        
        # Combine all areas
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        areas_combined = np.hstack(all_areas)
        
        total_time = time.time() - start_time
        
        logger.info(f"\n🎯 DATASET SUMMARY:")
        logger.info(f"   📊 Total sequences: {X_combined.shape[0]:,}")
        logger.info(f"   📐 Shape: {X_combined.shape}")
        logger.info(f"   📈 Classes: {np.bincount(y_combined.astype(int))}")
        logger.info(f"   💾 Memory: {X_combined.nbytes / (1024**3):.2f} GB")
        logger.info(f"   ⏱️ Time: {total_time:.1f}s")
        
        return X_combined, y_combined, areas_combined

## Class Distribution Visualization
def visualize_class_distribution(y_train, y_val, y_test=None):
    """Visualize class distribution across train/val/test sets"""
    
    class_names = ['Normal', 'Warning', 'Fire']
    
    # Count classes in each set
    train_counts = np.bincount(y_train.astype(int), minlength=3)
    val_counts = np.bincount(y_val.astype(int), minlength=3)
    
    if y_test is not None:
        test_counts = np.bincount(y_test.astype(int), minlength=3)
    else:
        test_counts = np.zeros(3)
    
    # Convert to percentages
    train_pct = train_counts / train_counts.sum() * 100
    val_pct = val_counts / val_counts.sum() * 100
    
    if y_test is not None:
        test_pct = test_counts / test_counts.sum() * 100
    else:
        test_pct = np.zeros(3)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw counts
    x = np.arange(len(class_names))
    width = 0.25
    
    ax1.bar(x - width, train_counts, width, label='Train')
    ax1.bar(x, val_counts, width, label='Validation')
    
    if y_test is not None:
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
    if y_test is not None:
        for i, v in enumerate(test_counts):
            ax1.text(i + width, v + 0.1, f"{v:,}", ha='center')
    
    # Percentages
    ax2.bar(x - width, train_pct, width, label='Train')
    ax2.bar(x, val_pct, width, label='Validation')
    
    if y_test is not None:
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
    if y_test is not None:
        for i, v in enumerate(test_pct):
            ax2.text(i + width, v + 0.5, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Save figure if enabled
    if VISUALIZATION_CONFIG['save_figures']:
        plt.savefig(f"{VISUALIZATION_CONFIG['figure_dir']}/class_distribution.png", dpi=300)
    
    plt.show()

## Load and Prepare Dataset
# Initialize data loader
data_loader = OptimizedDataLoader()

# For demonstration purposes, create synthetic data instead of loading from S3
# In a real scenario, you would use: X, y, areas = data_loader.load_all_data_sample()
X, y, areas = create_synthetic_data(n_samples=50000, n_features=6, n_timesteps=60, n_areas=5)

# Create sequences
X_sequences = []
y_sequences = []
areas_sequences = []

for area_idx in range(5):  # 5 areas
    area_mask = areas == area_idx
    X_area = X[area_mask]
    y_area = y[area_mask]
    
    if len(X_area) > 60:  # Minimum sequence length
        X_seq, y_seq = data_loader.create_sequences(X_area, y_area, seq_len=60, step=10)
        areas_seq = np.full(len(X_seq), area_idx)
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        areas_sequences.append(areas_seq)

# Combine sequences from all areas
X = np.vstack(X_sequences)
y = np.hstack(y_sequences)
areas = np.hstack(areas_sequences)

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
visualize_class_distribution(y_train, y_val, y_test)