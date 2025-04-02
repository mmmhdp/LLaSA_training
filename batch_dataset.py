import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from os.path import join, exists
from collections import OrderedDict
import torchaudio
from torchaudio.transforms import Resample
from argparse import ArgumentParser
from time import time

# Import transformers components
from transformers import (
    AutoTokenizer, 
    AutoFeatureExtractor, 
    Wav2Vec2BertModel
)

# Import custom codec components
from vq.codec_encoder import CodecEncoder
from vq.codec_decoder_vocos import CodecDecoderVocos
from vq.module import SemanticDecoder, SemanticEncoder

# Import data handling components
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Audio
import torch.nn as nn
from typing import List, Tuple

def pad_audio_batch(batch):
    audio_list, feat_list, text_list, fname_list, audio_length = zip(*batch)
    feat_list = list(feat_list)
    
    max_length_feat = max([feat.shape[1] for feat in feat_list])
    max_length = max_length_feat * 320
    padded_audios = []
 
    for audio in audio_list:
        padding = max_length - audio.shape[1] 
        if padding > 0:
            padded_audio = F.pad(audio, (0, padding), mode='constant', value=0) 
        else:
            padded_audio = audio[:,:max_length]
        padded_audios.append(padded_audio)
    padded_audios = torch.stack(padded_audios)
    
    padded_feat_list = []
    for feat in feat_list:
        padding = max_length_feat - feat.shape[1]
        padded_feat = F.pad(feat, (0, 0, 0, padding), mode='constant', value=0)
        padded_feat_list.append(padded_feat)
 
    padded_feat_list = torch.stack(padded_feat_list)
    
    return padded_audios.float(), padded_feat_list.float(), text_list, fname_list, audio_length

class HuggingfaceDataset(Dataset):
    def __init__(
        self,
        dataset,
        sampling_rate,
        feature_extractor_name="facebook/w2v-bert-2.0",
        audio_norm_scale: float = 1.0,
    ):
        self.dataset = dataset
        self.sampling_rate = sampling_rate
        self.audio_norm_scale = audio_norm_scale
        self.hop_length = 320
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    def __getitem__(self, index):
        item = self.dataset[index]
        
        # Get audio from dataset
        audio_data = item['audio']
        audio = torch.tensor(audio_data['array'], dtype=torch.float32).unsqueeze(0)
        
        # Get transcript if available
        text = item.get('transcript', item.get('text', ''))
        
        # Resample if needed
        if audio_data['sampling_rate'] != self.sampling_rate:
            audio = Resample(audio_data['sampling_rate'], self.sampling_rate)(audio)
            
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale
 
        audio_pad = F.pad(audio, (160, 160))

        feat = self.feature_extractor(
            audio_pad,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).data['input_features']

        # Use item id or path as filename
        fname = str(item.get('id', f'sample_{index}'))
 
        return audio, feat, text, fname, int(audio.shape[1] / self.hop_length)
 
    def __len__(self):
        return len(self.dataset)

def save_processed_data(vq_codes, speech_ids, text_ids, sample_ids, output_dir, max_length, tokenizer, split="train"):
    """
    Save data in the split-specific format with two files per split
    """
    # Prepare memmap paths for this split
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'vq_codes'), exist_ok=True)
    
    memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
    shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")
    
    all_sequences = []
    saved_count = 0
    
    for i, sample_id in enumerate(sample_ids):
        try:
            # Save VQ codes
            code_path = os.path.join(output_dir, 'vq_codes', f'{sample_id}.npy')
            vq_code = vq_codes[i].numpy().astype(np.int32)  # No need for .cpu() as it's already on CPU
            np.save(code_path, vq_code)
            saved_count += 1
            
            # Create sequence with text and speech tokens
            text_sequence = text_ids[i]
            speech_sequence = speech_ids[i]
            
            # Calculate available space for text
            MAX_TEXT_SPACE = max_length - len(speech_sequence)
            if MAX_TEXT_SPACE < 0:
                print(f"Warning: Speech sequence too long ({len(speech_sequence)} tokens) for sample {sample_id}, skipping...")
                continue
                
            # Truncate text to fit
            truncated_text = text_sequence[:MAX_TEXT_SPACE]
            
            # Build final sequence
            final_sequence = (
                truncated_text +
                speech_sequence +
                [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_sequence))
            )[:max_length]
            
            all_sequences.append(final_sequence)
        except Exception as e:
            print(f"Error saving sample {sample_id}: {e}")
            continue
    
    print(f"Successfully saved {saved_count} VQ code files")
    
    # Save sequences to memmap
    if all_sequences:
        # Save to disk
        arr = np.memmap(memmap_path, dtype=np.int32, mode="w+", 
                      shape=(len(all_sequences), max_length))
        arr[:] = np.array(all_sequences, dtype=np.int32)
        arr.flush()
        np.save(shape_path, np.array([len(all_sequences), max_length]))
        
        print(f"\n=== {split} Split Summary ===")
        print(f"Saved {len(all_sequences)} sequences of length {max_length}")
        print(f"Memmap file size: {os.path.getsize(memmap_path)/1e6:.2f}MB")
        print(f"Shape: {np.load(shape_path)}")
    else:
        print(f"\nWarning: No valid sequences found for split {split}")

def get_speech_ids(vq_codes, tokenizer):
    """Convert VQ codes to speech token IDs"""
    batch_speech_ids = []
    
    for codes in vq_codes:
        try:
            # Convert codes to token IDs using the special speech tokens
            speech_ids = (
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")] +
                [tokenizer.convert_tokens_to_ids(f"<|s_{int(code)}|>") for code in codes] +
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
            )
            batch_speech_ids.append(speech_ids)
        except Exception as e:
            print(f"Error converting codes to speech IDs: {e}")
            # Provide a fallback (empty sequence with start/end tags)
            speech_ids = [
                tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>"),
                tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
            ]
            batch_speech_ids.append(speech_ids)
    
    return batch_speech_ids

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0, help='Local GPU device ID')
    parser.add_argument('--dataset-name', type=str, required=True, help='Huggingface dataset name')
    parser.add_argument('--dataset-config', type=str, default=None, help='Dataset configuration name if applicable')
    parser.add_argument('--dataset-split', type=str, default='train', help='Dataset split to use (train, validation, test)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--ckpt', type=str, default='./ckpt/epoch=4-step=1400000.ckpt', help='Path to the VQGAN model checkpoint')
    parser.add_argument('--tokenizer-name', type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct", help='Name of the tokenizer')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory for saving codes')
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker threads for the DataLoader')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs to divide workload across')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length for output')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate for audio processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process fewer samples)')
    parser.add_argument('--subset-ratio', type=float, default=1.0, help='Ratio of dataset to process (0.0-1.0)')
    parser.add_argument('--process-all-splits', action='store_true', help='Process all available splits instead of just the specified one')
 
    device_id = int(os.getenv('LOCAL_RANK', 0))  
    args = parser.parse_args()
    sr = args.sample_rate

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer with special tokens
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Add special tokens for text/speech
    special_tokens = [
        "<|TEXT_GENERATION_START|>",
        "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>",
        "<|SPEECH_UNDERSTANDING_END|>",
    ]
    
    # Add speech tokens (0-65535)
    speech_tokens = [f"<|s_{i}|>" for i in range(65536)]
    all_tokens = special_tokens + speech_tokens
    
    tokenizer.add_tokens(all_tokens)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 2
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Device setup
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    # Load VQGAN model
    print(f'Loading VQGAN checkpoint from {args.ckpt}')
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        ckpt = ckpt['state_dict']

        # Filter state dict for different components
        filtered_state_dict_codec = OrderedDict()
        filtered_state_dict_semantic_encoder = OrderedDict()
        filtered_state_dict_gen = OrderedDict()
        filtered_state_dict_fc_post_a = OrderedDict()
        filtered_state_dict_fc_prior = OrderedDict()

        for key, value in ckpt.items():
            if key.startswith('CodecEnc.'):
                new_key = key[len('CodecEnc.'):]
                filtered_state_dict_codec[new_key] = value
            elif key.startswith('generator.'):
                new_key = key[len('generator.'):]
                filtered_state_dict_gen[new_key] = value
            elif key.startswith('fc_post_a.'):
                new_key = key[len('fc_post_a.'):]
                filtered_state_dict_fc_post_a[new_key] = value
            elif key.startswith('SemanticEncoder_module.'):
                new_key = key[len('SemanticEncoder_module.'):]
                filtered_state_dict_semantic_encoder[new_key] = value
            elif key.startswith('fc_prior.'):
                new_key = key[len('fc_prior.'):]
                filtered_state_dict_fc_prior[new_key] = value

        # Initialize models
        semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)
        semantic_model.eval()

        SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        SemanticEncoder_module.load_state_dict(filtered_state_dict_semantic_encoder)
        SemanticEncoder_module.eval()

        encoder = CodecEncoder()
        encoder.load_state_dict(filtered_state_dict_codec)
        encoder.eval()

        decoder = CodecDecoderVocos()
        decoder.load_state_dict(filtered_state_dict_gen)
        decoder.eval()

        fc_post_a = nn.Linear(2048, 1024)
        fc_post_a.load_state_dict(filtered_state_dict_fc_post_a)
        fc_post_a.eval()

        fc_prior = nn.Linear(2048, 2048)
        fc_prior.load_state_dict(filtered_state_dict_fc_prior)
        fc_prior.eval()

        # Move models to device
        semantic_model.to(device)
        SemanticEncoder_module.to(device)
        encoder.to(device)
        decoder.to(device)
        fc_post_a.to(device)
        fc_prior.to(device)
    except Exception as e:
        print(f"Error loading VQGAN model: {e}")
        raise Exception("Failed to load VQGAN model")

    # Load dataset from Huggingface
    print(f"Loading dataset: {args.dataset_name}")
    
    # First load to get available splits
    full_dataset = load_dataset(
        args.dataset_name, 
        args.dataset_config,
        streaming=False
    )
    
    # Get available splits
    available_splits = list(full_dataset.keys())
    print(f"Found splits: {available_splits}")
    
    # Determine which splits to process
    # splits_to_process = available_splits if args.process_all_splits else [args.dataset_split]
    
    for current_split in ["train"]:
        print(f"\nProcessing split: {current_split}")
        
        # Load the specific split
        dataset = full_dataset[current_split]
        
        # Apply debug or subset ratio if applicable
        if args.debug:
            print("\n*** DEBUG MODE ACTIVATED - PROCESSING FEWER SAMPLES ***\n")
            dataset = dataset.select(range(min(10, len(dataset))))
        elif args.subset_ratio < 1.0:
            if not 0.0 < args.subset_ratio <= 1.0:
                raise ValueError("subset_ratio must be between 0.0 and 1.0")
            num_samples = int(len(dataset) * args.subset_ratio)
            dataset = dataset.select(range(num_samples))
            print(f"Using {args.subset_ratio:.2%} of data: {num_samples} samples")
        elif args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        
        # Cast to Audio format with proper sampling rate
        dataset = dataset.cast_column("audio", Audio(sampling_rate=sr))
        
        # Split dataset across GPUs if using multiple
        if args.num_gpus > 1:
            # Calculate number of samples per GPU
            samples_per_gpu = len(dataset) // args.num_gpus
            # Calculate start and end indices for this GPU
            start_idx = device_id * samples_per_gpu
            end_idx = start_idx + samples_per_gpu if device_id < args.num_gpus - 1 else len(dataset)
            # Select subset for this GPU
            dataset = dataset.select(range(start_idx, end_idx))
        
        print(f"GPU {device_id} processing {len(dataset)} samples for split '{current_split}'")

        # Create dataset and dataloader
        hf_dataset = HuggingfaceDataset(
            dataset=dataset,
            sampling_rate=sr,
        )
        
        dataloader = DataLoader(
            hf_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=pad_audio_batch
        )

        st = time()
        all_vq_codes = []
        all_speech_ids = []
        all_text_ids = []
        all_sample_ids = []
        
        for batch in tqdm(dataloader, desc=f"Processing {current_split} on GPU {device_id}"):
            wavs, feats, texts, batch_sample_ids, lengths = batch
            wavs = wavs.to(device)

            with torch.no_grad():
                # Process batch of texts
                batch_text_ids = []
                for text in texts:
                    text_with_tags = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
                    text_ids = tokenizer.encode(text_with_tags, add_special_tokens=False)
                    batch_text_ids.append(text_ids)
                
                try:
                    # Extract features using VQGAN architecture
                    vq_emb = encoder(wavs)
                    vq_emb = vq_emb.transpose(1, 2)

                    semantic_target = semantic_model(feats[:,0,:,:].to(device))
                    semantic_target = semantic_target.hidden_states[16]
                    semantic_target = semantic_target.transpose(1, 2)
                    semantic_target = SemanticEncoder_module(semantic_target)

                    vq_emb = torch.cat([semantic_target, vq_emb], dim=1)
                    vq_emb = fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

                    _, vq_code, _ = decoder(vq_emb, vq=True)
                    
                    # Get speech token IDs from VQ codes
                    batch_speech_ids = get_speech_ids(vq_code[:, 0, :].cpu().numpy(), tokenizer)
                    
                    # Store the results for this batch
                    all_vq_codes.extend([vc for vc in vq_code[:, 0, :].detach().cpu()])
                    all_speech_ids.extend(batch_speech_ids)
                    all_text_ids.extend(batch_text_ids)
                    all_sample_ids.extend(batch_sample_ids)
                        
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue

        # Convert list of tensors to a single tensor for VQ codes
        if all_vq_codes:
            # Save the processed data for this batch
            print(f"Processed {len(all_vq_codes)} samples, now saving...")
            save_processed_data(
                all_vq_codes, 
                all_speech_ids, 
                all_text_ids, 
                all_sample_ids, 
                args.output_dir, 
                args.max_length, 
                tokenizer,
                split=current_split
            )
        else:
            print(f"No valid samples processed for split {current_split}")

        et = time()
        print(f'Finished processing {current_split} on GPU {device_id}, processing time: {(et - st)/60:.2f} mins')