"""
IoT Data Loader for the new area-based fire detection system.
Handles loading and preprocessing of synthetic datasets for different areas.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class IoTFireDataset(Dataset):
    """Dataset for IoT-based fire detection with area-specific sensors."""
    
    def __init__(self, data_dir: str, sequence_length: int = 60, 
                 overlap: float = 0.5, normalize: bool = True):
        """
        Initialize the IoT fire detection dataset.
        
        Args:
            data_dir (str): Directory containing the CSV files
            sequence_length (int): Length of time sequences
            overlap (float): Overlap between sequences (0.0 to 1.0)
            normalize (bool): Whether to normalize the data
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.normalize = normalize
        
        # Area configuration matching the sensor types
        self.area_config = {
            'kitchen': {
                'file': 'voc_data.csv',
                'features': ['value'],
                'sensor_id': 'kitchen_voc'
            },
            'electrical': {
                'file': 'arc_data.csv', 
                'features': ['value'],
                'sensor_id': 'electrical_panel_arc'
            },
            'laundry_hvac': {
                'file': 'laundry_data.csv',
                'features': ['temperature', 'current'],
                'sensor_id': 'laundry_room'
            },
            'living_bedroom': {
                'file': 'asd_data.csv',
                'features': ['value'],
                'sensor_id': 'living_room_asd'
            },
            'basement_storage': {
                'file': 'basement_data.csv',
                'features': ['temperature', 'humidity', 'gas_levels'],
                'sensor_id': 'basement_iot'
            }
        }
        
        # Load and preprocess data
        self.area_data = {}
        self.sequences = []
        self.labels = []
        
        self._load_data()
        self._create_sequences()
        
        logger.info(f"IoT Dataset loaded: {len(self.sequences)} sequences")
    
    def _load_data(self):
        """Load data from CSV files for each area."""
        for area_name, config in self.area_config.items():
            file_path = self.data_dir / config['file']
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
            
            logger.info(f"Loading {area_name} data from {file_path}")
            
            # Load data in chunks to handle large files
            chunk_size = 100000
            chunks = []
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Filter by sensor_id if needed
                if 'sensor_id' in chunk.columns:
                    chunk = chunk[chunk['sensor_id'] == config['sensor_id']]
                
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Extract features
            feature_data = df[config['features']].values.astype(np.float32)
            
            # Extract labels (anomaly detection)
            labels = df['is_anomaly'].values.astype(bool)
            
            # Store processed data
            self.area_data[area_name] = {
                'features': feature_data,
                'labels': labels,
                'timestamps': df['timestamp'].values
            }
            
            logger.info(f"{area_name}: {len(feature_data)} samples, "
                       f"{labels.sum()} anomalies ({labels.mean():.2%})")
    
    def _create_sequences(self):
        """Create time sequences from the loaded data."""
        # Find the minimum length across all areas
        min_length = min(len(data['features']) for data in self.area_data.values())
        
        # Calculate step size based on overlap
        step_size = int(self.sequence_length * (1 - self.overlap))
        
        # Create sequences
        for start_idx in range(0, min_length - self.sequence_length + 1, step_size):
            end_idx = start_idx + self.sequence_length
            
            sequence_data = {}
            sequence_labels = []
            
            for area_name, data in self.area_data.items():
                # Extract sequence for this area
                area_sequence = data['features'][start_idx:end_idx]
                area_labels = data['labels'][start_idx:end_idx]
                
                sequence_data[area_name] = torch.tensor(area_sequence, dtype=torch.float32)
                sequence_labels.append(area_labels.any())  # Any anomaly in sequence
            
            # Determine overall sequence label and lead time
            has_anomaly = any(sequence_labels)
            lead_time_category = self._determine_lead_time_category(sequence_data)
            
            self.sequences.append(sequence_data)
            self.labels.append({
                'has_anomaly': has_anomaly,
                'lead_time_category': lead_time_category,
                'area_anomalies': {area: label for area, label in 
                                 zip(self.area_data.keys(), sequence_labels)}
            })
        
        # Normalize data if requested
        if self.normalize:
            self._normalize_sequences()
    
    def _determine_lead_time_category(self, sequence_data: Dict[str, torch.Tensor]) -> int:
        """
        Determine lead time category based on area and anomaly patterns.
        
        Returns:
            int: 0=immediate, 1=hours, 2=days, 3=weeks
        """
        # Simple heuristic based on area types and anomaly severity
        # In practice, this would be based on domain knowledge and historical data
        
        # Check for immediate risks (high values in critical areas)
        if 'kitchen' in sequence_data:
            voc_max = sequence_data['kitchen'].max()
            if voc_max > 200:  # High VOC levels
                return 0  # Immediate
        
        if 'living_bedroom' in sequence_data:
            asd_max = sequence_data['living_bedroom'].max()
            if asd_max > 10:  # High particle levels
                return 0  # Immediate
        
        # Check for hours-level risks
        if 'laundry_hvac' in sequence_data:
            temp_trend = sequence_data['laundry_hvac'][:, 0]  # Temperature
            if temp_trend[-1] - temp_trend[0] > 10:  # Rising temperature
                return 1  # Hours
        
        if 'basement_storage' in sequence_data:
            gas_levels = sequence_data['basement_storage'][:, 2]  # Gas levels
            if gas_levels.max() > 15:
                return 1  # Hours
        
        # Check for days-level risks (electrical)
        if 'electrical' in sequence_data:
            arc_count = sequence_data['electrical'].sum()
            if arc_count > 5:  # Multiple arc events
                return 2  # Days
        
        # Default to weeks for gradual degradation
        return 3  # Weeks
    
    def _normalize_sequences(self):
        """Normalize sequences using z-score normalization per area."""
        # Calculate statistics across all sequences
        area_stats = {}
        
        for area_name in self.area_data.keys():
            all_values = torch.cat([seq[area_name].flatten() for seq in self.sequences])
            area_stats[area_name] = {
                'mean': all_values.mean(),
                'std': all_values.std() + 1e-8  # Add small epsilon to avoid division by zero
            }
        
        # Apply normalization
        for sequence in self.sequences:
            for area_name in sequence.keys():
                stats = area_stats[area_name]
                sequence[area_name] = (sequence[area_name] - stats['mean']) / stats['std']
        
        self.normalization_stats = area_stats
        logger.info("Data normalization completed")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Get a sequence and its labels.
        
        Returns:
            Tuple containing:
                - Dict of area-specific sequences
                - Dict of labels and metadata
        """
        return self.sequences[idx], self.labels[idx]
    
    def get_area_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each area."""
        stats = {}
        
        for area_name, data in self.area_data.items():
            features = data['features']
            stats[area_name] = {
                'mean': float(features.mean()),
                'std': float(features.std()),
                'min': float(features.min()),
                'max': float(features.max()),
                'anomaly_rate': float(data['labels'].mean())
            }
        
        return stats
    
    def create_balanced_subset(self, max_samples: int = 10000) -> 'IoTFireDataset':
        """Create a balanced subset of the dataset."""
        # Separate sequences by lead time category
        categories = {0: [], 1: [], 2: [], 3: []}
        
        for i, label in enumerate(self.labels):
            categories[label['lead_time_category']].append(i)
        
        # Sample equally from each category
        samples_per_category = max_samples // 4
        selected_indices = []
        
        for category_indices in categories.values():
            if len(category_indices) > samples_per_category:
                selected = np.random.choice(category_indices, samples_per_category, replace=False)
            else:
                selected = category_indices
            selected_indices.extend(selected)
        
        # Create subset
        subset = IoTFireDataset.__new__(IoTFireDataset)
        subset.data_dir = self.data_dir
        subset.sequence_length = self.sequence_length
        subset.overlap = self.overlap
        subset.normalize = self.normalize
        subset.area_config = self.area_config
        subset.area_data = self.area_data
        
        subset.sequences = [self.sequences[i] for i in selected_indices]
        subset.labels = [self.labels[i] for i in selected_indices]
        
        if hasattr(self, 'normalization_stats'):
            subset.normalization_stats = self.normalization_stats
        
        logger.info(f"Created balanced subset: {len(subset.sequences)} samples")
        return subset


def create_iot_dataloaders(data_dir: str, batch_size: int = 32, 
                          sequence_length: int = 60, train_split: float = 0.7,
                          val_split: float = 0.15, max_samples: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for IoT fire detection.
    
    Args:
        data_dir (str): Directory containing CSV files
        batch_size (int): Batch size for dataloaders
        sequence_length (int): Length of time sequences
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        max_samples (int): Maximum number of samples to use (for testing)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = IoTFireDataset(data_dir, sequence_length=sequence_length)
    
    # Create balanced subset if requested
    if max_samples is not None:
        dataset = dataset.create_balanced_subset(max_samples)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    
    logger.info(f"Created dataloaders: Train={len(train_dataset)}, "
               f"Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def collate_iot_batch(batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Custom collate function for IoT fire detection batches.
    
    Args:
        batch: List of (sequence_dict, label_dict) tuples
        
    Returns:
        Tuple of (batched_sequences, batched_labels)
    """
    sequences, labels = zip(*batch)
    
    # Batch area sequences
    batched_sequences = {}
    area_names = sequences[0].keys()
    
    for area_name in area_names:
        area_sequences = [seq[area_name] for seq in sequences]
        batched_sequences[area_name] = torch.stack(area_sequences, dim=0)
    
    # Batch labels
    batched_labels = {
        'has_anomaly': torch.tensor([label['has_anomaly'] for label in labels], dtype=torch.bool),
        'lead_time_category': torch.tensor([label['lead_time_category'] for label in labels], dtype=torch.long)
    }
    
    # Batch area-specific anomalies
    for area_name in area_names:
        area_anomalies = [label['area_anomalies'][area_name] for label in labels]
        batched_labels[f'{area_name}_anomaly'] = torch.tensor(area_anomalies, dtype=torch.bool)
    
    return batched_sequences, batched_labels