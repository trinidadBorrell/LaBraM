#!/usr/bin/env python3
"""
LaBraM Inference Script for EEG Data Reconstruction

This script reads .fif epoch files, performs inference with LaBraM model,
and saves reconstructed data back to .fif format.

Usage:
    python inference.py --subject_id <ID> [--checkpoint_path <path>]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
import mne

# Import LaBraM model
#from modeling_pretrain import NeuralTransformerForMEM
#from modeling_finetune import NeuralTransformer
from modeling_vqnsp import VQNSP
from timm.models import create_model
import utils

# Import utility functions from our custom utils module
import sys
sys.path.insert(0, '/home/triniborrell/home/projects')
import importlib.util
spec = importlib.util.spec_from_file_location("eeg_utils", "/home/triniborrell/home/projects/utils.py")
eeg_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eeg_utils)

# Import the functions we need
read_fif_data = eeg_utils.read_fif_data
select_electrodes = eeg_utils.select_electrodes
resample_data = eeg_utils.resample_data
get_standard_19_electrodes = eeg_utils.get_standard_19_electrodes
preprocess_eeg_pipeline = eeg_utils.preprocess_eeg_pipeline


def load_vqnsp_model(tokenizer_path):
    """Load VQNSP tokenizer model for proper encoder-decoder pipeline."""
    print(f"Loading VQNSP tokenizer from {tokenizer_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # Create VQNSP model using the registered model function
    # Use code_dim=64 to match the checkpoint dimensions
    model = create_model(
        'vqnsp_encoder_base_decoder_3x200x12',
        pretrained=True,
        pretrained_weight=tokenizer_path,
        as_tokenzer=True,
        n_code=8192,
        code_dim=64,
        EEG_size=1600
    )
    
    model.to(device)
    model.eval()
    print("VQNSP model loaded successfully")
    return model

'''
def load_labram_model(checkpoint_path, device='cuda', model_type='pretrained'):
    """Load LaBraM model from checkpoint."""
    print(f"Loading LaBraM model from {checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    if model_type == 'pretrained':
        # Initialize pretraining model (for labram-base checkpoints)
        from functools import partial
        import torch.nn as nn
        model = NeuralTransformerForMEM(
            EEG_size=1600,
            patch_size=200,
            in_chans=1,
            out_chans=8,
            vocab_size=8192,
            embed_dim=200,
            depth=12,
            num_heads=10,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1
        )
    else:
        # Initialize finetuning model
        model = NeuralTransformer(
            EEG_size=1600,
            patch_size=200,
            in_chans=1,
            out_chans=8,
            num_classes=0,  # Set to 0 for feature extraction
            embed_dim=200,
            depth=12,
            num_heads=10,
            use_mean_pooling=True
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    checkpoint_model = None
    for model_key in ['model', 'module']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f"Load state_dict by model_key = {model_key}")
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    
    # Remove head weights if they exist (for feature extraction)
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    # Remove relative position index keys
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
    
    # Load state dict with proper handling for pretraining vs finetuning models
    if model_type == 'pretrained':
        # For pretraining models, we need to extract the student model
        utils.load_state_dict(model, checkpoint_model, prefix='')
        # Use the student model for inference (feature extraction)
        model = model.student
    else:
        utils.load_state_dict(model, checkpoint_model, prefix='')
    
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    return model
'''

def load_and_preprocess_eeg(file_path, electrode_names=None, target_sfreq=None, use_standard_19=False):
    """
    Load and preprocess EEG data using utility functions.
    
    Args:
        file_path: Path to the FIF file
        electrode_names: List of electrode names to select (optional)
        target_sfreq: Target sampling frequency for resampling (optional)
        use_standard_19: Whether to use standard 19-electrode montage
    
    Returns:
        tuple: (processed_epochs, metadata)
    """
    print(f"Loading and preprocessing EEG data from: {file_path}")
    
    # Use the preprocessing pipeline from utils
    processed_epochs, metadata = preprocess_eeg_pipeline(
        file_path=file_path,
        electrode_names=electrode_names,
        target_sfreq=target_sfreq,
        use_standard_19=use_standard_19
    )
    
    return processed_epochs, metadata


def preprocess_eeg_data(eeg_data, max_channels=256):
    """
    Preprocess EEG data according to LaBraM requirements.
    
    Args:
        eeg_data: numpy array of shape (epochs, channels, timepoints)
        target_fs: target sampling frequency (default: 200 Hz)
        max_channels: maximum number of channels to keep (default: 128)
    
    Returns:
        Preprocessed EEG data ready for LaBraM inference
    """
    print(f"Preprocessing EEG data: {eeg_data.shape}")
    
    # Convert to microvolts (assuming data is in volts)
    # Note: This might need adjustment based on your data units
    eeg_data = eeg_data * 1e6  # Convert to microvolts
    
    # Divide by 100 as done in LaBraM training
    eeg_data = eeg_data / 100.0

    # Reduce channels if necessary to fit model's token limit
    if eeg_data.shape[1] > max_channels:
        print(f"Reducing channels from {eeg_data.shape[1]} to {max_channels}")
        # Option 1: Take first max_channels channels
        eeg_data = eeg_data[:, :max_channels, :]
        
    # Reshape to (epochs, channels, patches, patch_size)
    eeg_data = reshape_eeg_data(eeg_data, patch_size=200)
    
    return eeg_data


def reshape_eeg_data(eeg_data, patch_size=200, target_patches=8):
    """
    Reshape EEG data for LaBraM input format by grouping consecutive epochs.
    
    Instead of tiling data, this groups consecutive epochs together to form
    the required number of temporal patches. Each group of `target_patches` 
    epochs becomes one sample with `target_patches` time windows.
    
    Args:
        eeg_data: numpy array of shape (epochs, channels, timepoints)
        patch_size: size of each patch (default: 200)
        target_patches: number of patches required by model (default: 8)
    
    Returns:
        Reshaped data of format (n_groups, channels, target_patches, patch_size)
    """
    epochs, channels, timepoints = eeg_data.shape
    
    # Trim timepoints to patch_size (each epoch becomes one patch)
    if timepoints > patch_size:
        print(f"Trimming timepoints from {timepoints} to {patch_size}")
        eeg_data = eeg_data[:, :, :patch_size]
    elif timepoints < patch_size:
        print(f"Warning: timepoints ({timepoints}) < patch_size ({patch_size}). Padding with zeros.")
        pad_width = ((0, 0), (0, 0), (0, patch_size - timepoints))
        eeg_data = np.pad(eeg_data, pad_width, mode='constant', constant_values=0)
    
    # Group consecutive epochs to form target_patches time windows
    # Each group of target_patches epochs becomes one sample
    n_groups = epochs // target_patches
    remainder = epochs % target_patches
    
    if remainder > 0:
        print(f"Warning: {remainder} epochs will be dropped (not divisible by {target_patches})")
    
    # Trim to exact multiple of target_patches
    eeg_data = eeg_data[:n_groups * target_patches, :, :]
    
    # Reshape: (epochs, channels, patch_size) -> (n_groups, target_patches, channels, patch_size)
    # Then transpose to (n_groups, channels, target_patches, patch_size)
    reshaped_data = eeg_data.reshape(n_groups, target_patches, channels, patch_size)
    reshaped_data = reshaped_data.transpose(0, 2, 1, 3)  # (n_groups, channels, target_patches, patch_size)
    
    print(f"Grouped {epochs} epochs into {n_groups} samples of {target_patches} patches each")
    print(f"Reshaped data: ({epochs}, {channels}, {timepoints}) -> {reshaped_data.shape}")
    return reshaped_data


def perform_vqnsp_inference(vqnsp_model, eeg_data, device='cuda', batch_size=128, max_electrodes=128):
    """
    Perform VQNSP encoder-decoder inference on EEG data.
    
    Args:
        vqnsp_model: Loaded VQNSP model with encoder and decoder
        eeg_data: Preprocessed EEG data of shape (epochs, channels, patches, patch_size)
        device: Device for computation
        batch_size: Batch size for inference
        max_electrodes: Max electrodes per pass (128 * 8 patches = 1024 tokens limit)
    
    Returns:
        Reconstructed EEG data from encoder-decoder pipeline
    """
    print(f"Performing VQNSP inference on {eeg_data.shape[0]} epochs")
    
    n_epochs, num_electrodes = eeg_data.shape[0], eeg_data.shape[1]
    all_reconstructed = []
    
    with torch.no_grad():
        for num, i in enumerate(range(0, n_epochs, batch_size)):
            batch_end = min(i + batch_size, n_epochs)
            batch_data = eeg_data[i:batch_end]
            
            print('Performing VQNSP inference on batch ', num, 'of size ', batch_size)
            
            if num_electrodes > max_electrodes:
                mid = num_electrodes // 2
                batch_tensor_1 = torch.tensor(batch_data[:, :mid], dtype=torch.float32).to(device)
                batch_tensor_2 = torch.tensor(batch_data[:, mid:], dtype=torch.float32).to(device)
                quantize_1, _, _ = vqnsp_model.encode(batch_tensor_1)
                quantize_2, _, _ = vqnsp_model.encode(batch_tensor_2)
                rec_1, rec_angle_1 = vqnsp_model.decode(quantize_1)
                rec_2, rec_angle_2 = vqnsp_model.decode(quantize_2)
                rec, rec_angle = torch.cat([rec_1, rec_2], dim=1), torch.cat([rec_angle_1, rec_angle_2], dim=1)
            else:
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
                quantize, embed_ind, loss = vqnsp_model.encode(batch_tensor)
                rec, rec_angle = vqnsp_model.decode(quantize)
            
            # From fourier space to time domain
            x_fft_reconstructed = torch.polar(rec, rec_angle)
            x_reconstructed = torch.fft.ifft(x_fft_reconstructed, dim=-1)
            x_reconstructed = x_reconstructed.real

            # Move back to CPU and store
            all_reconstructed.append(x_reconstructed.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {batch_end}/{n_epochs} epochs")
    
    # Concatenate all reconstructed data
    all_reconstructed = np.concatenate(all_reconstructed, axis=0)
    print(f"VQNSP inference completed. Reconstructed shape: {all_reconstructed.shape}")
    
    return all_reconstructed

'''
def perform_inference(model, eeg_data, device='cuda', batch_size=32, max_electrodes=128):
    """
    Perform LaBraM inference on EEG data.
    
    Args:
        model: Loaded LaBraM model (student model from pretraining)
        eeg_data: Preprocessed EEG data of shape (epochs, channels, patches, patch_size)
        device: Device for computation
        batch_size: Batch size for inference
        max_electrodes: Maximum electrodes per inference pass (default: 128)
    
    Returns:
        Model features/embeddings (not frequency domain - these are learned representations)
    """
    print(f"Performing inference on {eeg_data.shape[0]} epochs")
    
    n_epochs, num_electrodes = eeg_data.shape[0], eeg_data.shape[1]
    all_features = []
    
    with torch.no_grad():
        for i in range(0, n_epochs, batch_size):
            batch_end = min(i + batch_size, n_epochs)
            batch_data = eeg_data[i:batch_end]
            
            if num_electrodes > max_electrodes:
                mid = num_electrodes // 2
                batch_tensor_1 = torch.tensor(batch_data[:, :mid], dtype=torch.float32).to(device)
                batch_tensor_2 = torch.tensor(batch_data[:, mid:], dtype=torch.float32).to(device)
                features_1 = model.forward_features(batch_tensor_1, return_all_tokens=True)
                features_2 = model.forward_features(batch_tensor_2, return_all_tokens=True)
                features = torch.cat([features_1, features_2], dim=1)
            else:
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
                features = model.forward_features(batch_tensor, return_all_tokens=True)
            
            all_features.append(features.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {batch_end}/{n_epochs} epochs")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    print(f"Inference completed. Features shape: {all_features.shape}")
    
    return all_features
'''


def save_reconstructed_data(reconstructed_data, original_epochs, output_path, n_channels=128, n_patches=8):
    """
    Save reconstructed EEG data.
    
    The VQNSP output has shape (n_groups, n_channels * n_patches, features).
    This represents reconstructed FFT features, not raw time-domain EEG.
    
    Args:
        reconstructed_data: Reconstructed data of shape (n_groups, n_tokens, features)
        original_epochs: Original MNE Epochs object for metadata
        output_path: Path to save the reconstructed data
        n_channels: Number of channels used in preprocessing (default: 128)
        n_patches: Number of patches per group (default: 8)
    """
    print(f"Saving reconstructed data to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    n_groups, n_tokens, features = reconstructed_data.shape
    print(f"Reconstructed data shape: {reconstructed_data.shape}")
    print(f"  - {n_groups} groups (each group = {n_patches} original epochs)")
    print(f"  - {n_tokens} tokens ({n_channels} channels × {n_patches} patches)")
    print(f"  - {features} features (FFT reconstruction dimension)")
    
    # Reshape to (n_groups, n_channels, n_patches, features)
    reshaped_data = reconstructed_data.reshape(n_groups, n_channels, n_patches, features)
    print(f"Reshaped to: {reshaped_data.shape}")
    
    # Save as numpy file
    np_output_path = os.path.join(output_path, 'vqnsp_reconstructed.npy')
    np.save(np_output_path, reshaped_data)
    print(f"Reconstructed data saved to: {np_output_path}")
    
    # Save metadata
    metadata = {
        'original_n_epochs': len(original_epochs),
        'n_groups': n_groups,
        'n_channels': n_channels,
        'n_patches': n_patches,
        'features': features,
        'original_sfreq': original_epochs.info['sfreq'],
        'channel_names': original_epochs.info['ch_names'][:n_channels] if len(original_epochs.info['ch_names']) >= n_channels else original_epochs.info['ch_names'],
        'description': 'VQNSP reconstructed FFT features. Shape: (n_groups, n_channels, n_patches, features)'
    }
    metadata_path = os.path.join(output_path, 'vqnsp_metadata.npy')
    np.save(metadata_path, metadata)
    print(f"Metadata saved to: {metadata_path}")
    
    print(f"Reconstructed data saved successfully")


def save_reconstructed_as_fif(reconstructed_data, original_epochs, output_path, n_channels=128, n_patches=8):
    """
    Convert grouped reconstructed data back to individual epochs and save as .fif.
    
    Transforms (n_groups, n_channels, n_patches, timepoints) to 
    (n_groups * n_patches, n_channels, timepoints) and saves as MNE Epochs.
    
    Args:
        reconstructed_data: Data of shape (n_groups, n_channels, n_patches, timepoints)
                           or (n_groups, n_tokens, features) which will be reshaped
        original_epochs: Original MNE Epochs object for metadata
        output_path: Path to save the .fif file
        n_channels: Number of channels used in preprocessing (default: 128)
        n_patches: Number of patches per group (default: 8)
    """
    print(f"Converting reconstructed data to .fif format...")
    
    # Handle different input shapes
    if reconstructed_data.ndim == 3:
        # Shape is (n_groups, n_tokens, features) - reshape to (n_groups, n_channels, n_patches, features)
        n_groups, n_tokens, features = reconstructed_data.shape
        reconstructed_data = reconstructed_data.reshape(n_groups, n_channels, n_patches, features)
    
    n_groups, n_ch, n_seq, timepoints = reconstructed_data.shape
    print(f"Input shape: ({n_groups}, {n_ch}, {n_seq}, {timepoints})")
    
    # Reshape: (n_groups, n_channels, n_patches, timepoints) -> (n_groups * n_patches, n_channels, timepoints)
    # First transpose to (n_groups, n_patches, n_channels, timepoints)
    data_transposed = reconstructed_data.transpose(0, 2, 1, 3)
    # Then reshape to (n_groups * n_patches, n_channels, timepoints)
    data_epochs = data_transposed.reshape(n_groups * n_seq, n_ch, timepoints)
    print(f"Reshaped to epochs: {data_epochs.shape}")
    
    # Convert back to volts
    # Preprocessing did: volts * 1e6 / 100 = data in units of 100 µV
    # So to reverse: data * 100 / 1e6 = volts
    data_epochs = data_epochs * 100.0 / 1e6
    print(f"Converted back to volts (V)")
    
    # Create new info object with the subset of channels used
    original_ch_names = original_epochs.info['ch_names']
    used_ch_names = original_ch_names[:n_channels] if len(original_ch_names) >= n_channels else original_ch_names
    
    # Pick only the channels we used from the original info
    new_info = original_epochs.info.copy()
    # Get indices of channels to keep
    ch_indices = [original_epochs.info['ch_names'].index(ch) for ch in used_ch_names if ch in original_epochs.info['ch_names']]
    new_info = mne.pick_info(new_info, ch_indices)
    
    print(f"Using {len(new_info['ch_names'])} channels: {new_info['ch_names'][:5]}...")
    
    # Create events for the new epochs
    n_new_epochs = n_groups * n_seq
    events = np.column_stack([
        np.arange(n_new_epochs) * timepoints,  # Sample indices
        np.zeros(n_new_epochs, dtype=int),      # Previous event
        np.ones(n_new_epochs, dtype=int)        # Event ID
    ])
    
    # Create EpochsArray
    reconstructed_epochs = mne.EpochsArray(
        data_epochs,
        new_info,
        events=events,
        tmin=original_epochs.tmin,
        event_id={'reconstructed': 1},
        verbose=False
    )
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save to .fif format
    reconstructed_epochs.save(output_path, overwrite=True)
    print(f"Saved {n_new_epochs} epochs to: {output_path}")
    print(f"  Shape: {data_epochs.shape}")
    print(f"  Channels: {len(new_info['ch_names'])}")
    print(f"  Timepoints: {timepoints}")
    
    return reconstructed_epochs


def main():
    parser = argparse.ArgumentParser(description='LaBraM EEG Inference')
    parser.add_argument('--subject_id', type=str, default = '002',
                       help='Subject ID (e.g., "001")')
  #  parser.add_argument('--checkpoint_path', type=str, 
  #                     default='/home/triniborrell/home/projects/LaBraM/checkpoints/labram-base.pth',
  #                     help='Path to LaBraM checkpoint')
    parser.add_argument('--device', type=str, default= 'cpu',
                       help='Device for computation (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--model_type', type=str, default='pretrained',
                       choices=['pretrained', 'finetuned'],
                       help='Type of model checkpoint (pretrained for labram-base, finetuned for task-specific models)')
    parser.add_argument('--use_vqnsp', default = True,
                       help='Use VQNSP encoder-decoder pipeline for reconstruction')
    parser.add_argument('--tokenizer_path', type=str, 
                       default='/home/triniborrell/home/projects/LaBraM/checkpoints/vqnsp.pth',
                       help='Path to VQNSP tokenizer checkpoint')
    parser.add_argument('--electrode_names', nargs='+', default=None,
                       help='Specific electrode names to select (e.g., --electrode_names Fp1 Fp2 F3)')
    parser.add_argument('--use_standard_19', action='store_true',
                       help='Use standard 19-electrode montage')
    parser.add_argument('--target_sfreq', type=float, default=250,
                       help='Target sampling frequency for resampling (Hz)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_path = f"/data/project/eeg_foundation/data/control/fifdata_256/nice_epochs_rs/sub-{args.subject_id}/ses-01/eeg/sub-{args.subject_id}_ses-01_task-rs_acq-01_epo.fif"
   # output_path = f"/data/project/eeg_foundation/data/LaBram/data_256/sub-{args.subject_id}/ses-01/eeg/sub-{args.subject_id}_ses-01_task-rs_acq-01_epo_recon.fif"
    output_path = '/home/triniborrell/home/data/LaBram'
    print(f"Processing subject: {args.subject_id}")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Check if checkpoint exists
    #if not os.path.exists(args.checkpoint_path):
    #    print(f"Error: Checkpoint file does not exist: {args.checkpoint_path}")
    #    sys.exit(1)
    
    try:
        # Load and preprocess EEG data using utils functions (resample, choose electrodes)
        print("Loading EEG data...")
        epochs_obj, metadata = load_and_preprocess_eeg(
            file_path=input_path,
            electrode_names=args.electrode_names,
            target_sfreq=args.target_sfreq,
            use_standard_19=args.use_standard_19
        )

        eeg_data = epochs_obj.get_data()  # Shape: (n_epochs, channels, timepoints)
        print(f"Loaded EEG data: {eeg_data.shape}")
        print(f"Processing metadata: {metadata}")
        
        # Use VQNSP encoder-decoder pipeline for proper reconstruction
        if not os.path.exists(args.tokenizer_path):
            print(f"Error: VQNSP tokenizer file does not exist: {args.tokenizer_path}")
            sys.exit(1)
            
        # Load VQNSP model
        vqnsp_model = load_vqnsp_model(args.tokenizer_path)
            
        # Preprocess data --> Shape as desired: (batch_size, electrodes, samples, time_points)
        preprocessed_data = preprocess_eeg_data(eeg_data)
                        
        # Perform VQNSP inference (encode -> quantize -> decode)
        reconstructed_data = perform_vqnsp_inference(vqnsp_model, preprocessed_data, args.device, args.batch_size)
            
        # Reshape back to original format
        n_epochs, n_channels, n_timepoints = eeg_data.shape
        # VQNSP outputs (batch, seq_len, features) - need to reshape appropriately
        print(f"VQNSP output shape: {reconstructed_data.shape}")
            
        # Reshape VQNSP output back to original epoch format
        # This assumes VQNSP output needs to be reshaped to match original data dimensions
        if reconstructed_data.shape != eeg_data.shape:
            print(f"Reshaping VQNSP output from {reconstructed_data.shape} to match original ({n_epochs}, {n_channels}, {n_timepoints})")            
           
        # Save reconstructed data 
        # Save as numpy (intermediate format)
        save_reconstructed_data(reconstructed_data, epochs_obj, output_path)
            
        # Also save as .fif with proper epoch structure
        fif_output_path = os.path.join(output_path, 'vqnsp_reconstructed_epo.fif')
        save_reconstructed_as_fif(reconstructed_data, epochs_obj, fif_output_path)

        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
