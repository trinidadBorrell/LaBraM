#!/usr/bin/env python3
"""
LaBraM Batch Inference Script for EEG Data Reconstruction

This script finds all .fif epoch files in a directory structure,
performs inference with VQNSP model, and saves reconstructed data.

Usage:
    python batch_inference.py --base_dir <path> [--output_dir <path>]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import mne

# Import VQNSP model
from modeling_vqnsp import VQNSP
from timm.models import create_model
import utils

# Import utility functions from our custom utils module
sys.path.insert(0, '/home/triniborrell/home/projects')
import importlib.util
spec = importlib.util.spec_from_file_location("eeg_utils", "/home/triniborrell/home/projects/utils.py")
eeg_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eeg_utils)

# Import the functions we need
preprocess_eeg_pipeline = eeg_utils.preprocess_eeg_pipeline


def find_all_fif_files(base_dir):
    """
    Find all FIF files in the directory structure.
    
    Args:
        base_dir (str): Base directory containing subject folders
        
    Returns:
        list: List of tuples (subject_id, session_num, file_path)
    """
    fif_files = []
    base_path = Path(base_dir)
    
    # Find all subjects
    for subject_dir in base_path.glob("sub-*"):
        subject_id = subject_dir.name
        
        # Find all sessions
        for session_dir in subject_dir.glob("ses-*"):
            session_num = session_dir.name.split("-")[1]
            
            # Find FIF files
            for fif_file in session_dir.glob("eeg/sub-*_ses-*_task-rs_acq-01_epo.fif"):
                fif_files.append((subject_id, session_num, str(fif_file)))
    
    return sorted(fif_files)


def load_vqnsp_model(tokenizer_path):
    """Load VQNSP tokenizer model for proper encoder-decoder pipeline."""
    print(f"Loading VQNSP tokenizer from {tokenizer_path}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    # Create VQNSP model using the registered model function
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
        max_channels: maximum number of channels to keep (default: 256)
    
    Returns:
        Preprocessed EEG data ready for LaBraM inference
    """
    print(f"Preprocessing EEG data: {eeg_data.shape}")
    
    # Convert to microvolts (assuming data is in volts)
    eeg_data = eeg_data * 1e6  # Convert to microvolts
    
    # Divide by 100 as done in LaBraM training
    eeg_data = eeg_data / 100.0

    # Reduce channels if necessary to fit model's token limit
    if eeg_data.shape[1] > max_channels:
        print(f"Reducing channels from {eeg_data.shape[1]} to {max_channels}")
        eeg_data = eeg_data[:, :max_channels, :]
        
    # Reshape to (epochs, channels, patches, patch_size)
    eeg_data = reshape_eeg_data(eeg_data, patch_size=200)
    
    return eeg_data


def reshape_eeg_data(eeg_data, patch_size=200, target_patches=8):
    """
    Reshape EEG data for LaBraM input format by grouping consecutive epochs.
    
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
            
            print(f'Performing VQNSP inference on batch {num} of size {batch_size}')
            
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


def process_single_file(fif_path, subject_id, session_num, vqnsp_model, output_base_dir, args):
    """
    Process a single FIF file and save the reconstructed output.
    
    Args:
        fif_path: Path to the input FIF file
        subject_id: Subject ID (e.g., "sub-001")
        session_num: Session number (e.g., "01")
        vqnsp_model: Loaded VQNSP model
        output_base_dir: Base output directory
        args: Command line arguments
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {subject_id}, session {session_num}")
        print(f"Input: {fif_path}")
        
        # Load and preprocess EEG data
        epochs_obj, metadata = load_and_preprocess_eeg(
            file_path=fif_path,
            electrode_names=args.electrode_names,
            target_sfreq=args.target_sfreq,
            use_standard_19=args.use_standard_19
        )
        
        eeg_data = epochs_obj.get_data()  # Shape: (n_epochs, channels, timepoints)
        print(f"Loaded EEG data: {eeg_data.shape}")
        
        # Preprocess data
        preprocessed_data = preprocess_eeg_data(eeg_data)
        
        # Perform VQNSP inference
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reconstructed_data = perform_vqnsp_inference(
            vqnsp_model, preprocessed_data, device, args.batch_size
        )
        
        print(f"VQNSP output shape: {reconstructed_data.shape}")
        
        # Create output path: /output_base_dir/sub-{ID}/ses-{num}/{subject_id}_{session_num}_task-rs_acq-01_epo_recon.fif
        output_dir = os.path.join(output_base_dir, subject_id, f"ses-{session_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"{subject_id}_ses-{session_num}_task-rs_acq-01_epo_recon.fif"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save reconstructed data as .fif
        save_reconstructed_as_fif(reconstructed_data, epochs_obj, output_path)
        
        print(f"Successfully saved: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {fif_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='LaBraM Batch EEG Inference')
    parser.add_argument('--base_dir', type=str, 
                       default='/data/project/eeg_foundation/data/control/fifdata_256/nice_epochs_rs',
                       help='Base directory containing subject folders with FIF files')
    parser.add_argument('--output_dir', type=str,
                       default='/data/project/eeg_foundation/data/LaBram/fif_data_control_rs',
                       help='Output directory for reconstructed files')
    parser.add_argument('--tokenizer_path', type=str, 
                       default='/home/triniborrell/home/projects/LaBraM/checkpoints/vqnsp.pth',
                       help='Path to VQNSP tokenizer checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--electrode_names', nargs='+', default=None,
                       help='Specific electrode names to select')
    parser.add_argument('--use_standard_19', action='store_true',
                       help='Use standard 19-electrode montage')
    parser.add_argument('--target_sfreq', type=float, default=250,
                       help='Target sampling frequency for resampling (Hz)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LaBraM Batch Inference")
    print("="*60)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    
    # Check if base directory exists
    if not os.path.exists(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}")
        sys.exit(1)
    
    # Check if tokenizer exists
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: VQNSP tokenizer file does not exist: {args.tokenizer_path}")
        sys.exit(1)
    
    # Find all FIF files
    print("\nSearching for FIF files...")
    fif_files = find_all_fif_files(args.base_dir)
    
    if not fif_files:
        print("No FIF files found matching the pattern.")
        sys.exit(1)
    
    print(f"Found {len(fif_files)} FIF files to process:")
    for subject_id, session_num, fif_path in fif_files[:5]:
        print(f"  - {subject_id}, ses-{session_num}: {fif_path}")
    if len(fif_files) > 5:
        print(f"  ... and {len(fif_files) - 5} more")
    
    # Load VQNSP model once
    print("\nLoading VQNSP model...")
    vqnsp_model = load_vqnsp_model(args.tokenizer_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all files
    successful = 0
    failed = 0
    failed_files = []
    
    for idx, (subject_id, session_num, fif_path) in enumerate(fif_files):
        print(f"\n[{idx+1}/{len(fif_files)}] Processing {subject_id}, ses-{session_num}")
        
        success = process_single_file(
            fif_path, subject_id, session_num, 
            vqnsp_model, args.output_dir, args
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_files.append((subject_id, session_num, fif_path))
    
    # Summary
    print("\n" + "="*60)
    print("BATCH INFERENCE COMPLETE")
    print("="*60)
    print(f"Total files: {len(fif_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed_files:
        print("\nFailed files:")
        for subject_id, session_num, fif_path in failed_files:
            print(f"  - {subject_id}, ses-{session_num}: {fif_path}")
    
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
