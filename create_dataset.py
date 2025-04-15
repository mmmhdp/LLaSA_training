import os
import numpy as np
from datasets import load_dataset, load_from_disk
import torch
import torchaudio
from transformers import AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model
from tqdm import tqdm
import argparse

def preprocess_dataset(
    dataset_name: str,
    output_dir: str,
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    xcodec2_model_name: str = "HKUST-Audio/xcodec2",
    sample_rate: int = 16000,
    max_length: int = 2048,
    debug: bool = False,
    subset_ratio: float = 1.0
):
    # Load dataset
    print("Loading dataset...")
    #dataset = load_dataset(dataset_name)
    dataset = load_from_disk(dataset_name)
    
    # Get available splits
    splits = dataset.keys()
    print(f"Found splits: {splits}")

    if debug:
        print("\n*** DEBUG MODE ACTIVATED - PROCESSING 10 SAMPLES PER SPLIT ***\n")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if debug:
        print("Loaded tokenizer:", tokenizer)
        print("Original vocabulary size:", len(tokenizer))

    # Add special tokens
    Start_End_tokens = [
        "<|TEXT_GENERATION_START|>",
        "<|TEXT_GENERATION_END|>",
        "<|TEXT_UNDERSTANDING_START|>",
        "<|TEXT_UNDERSTANDING_END|>",
        "<|SPEECH_GENERATION_START|>",
        "<|SPEECH_GENERATION_END|>",
        "<|SPEECH_UNDERSTANDING_START|>",
        "<|SPEECH_UNDERSTANDING_END|>",
    ]

    new_speech_tokens = [f"<|s_{i}|>" for i in range(65536)]
    all_new_tokens = Start_End_tokens + new_speech_tokens
    num_added_tokens = tokenizer.add_tokens(all_new_tokens)
    tokenizer.pad_token_id = 2
    
    print(f"\nAdded {num_added_tokens} special tokens")
    print("New vocabulary size:", len(tokenizer))
    print("Pad token:", tokenizer.pad_token, "ID:", tokenizer.pad_token_id)

    # Load codec model
    codec_model = XCodec2Model.from_pretrained(xcodec2_model_name).eval().cuda()
    if debug:
        print("\nLoaded XCodec2 model:", codec_model.__class__.__name__)

    # Process each split
    for split in splits:
        print(f"\nProcessing split: {split}")
        
        # Get split data
        split_data = dataset[split]
        
        # Apply subset ratio if less than 1.0
        if subset_ratio < 1.0:
            num_samples = int(len(split_data) * subset_ratio)
            split_data = split_data.select(range(num_samples))
            print(f"Using {subset_ratio:.2%} of data: {num_samples} samples")
        elif debug:
            split_data = split_data.select(range(min(10, len(split_data))))
            print("Debug mode: using 10 samples")

        # Prepare memmap
        os.makedirs(output_dir, exist_ok=True)
        memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
        shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")

        all_sequences = []
        for idx, example in tqdm(enumerate(split_data), total=len(split_data)):
            # Process text
            text = f"<|TEXT_UNDERSTANDING_START|>{example['transcript']}<|TEXT_UNDERSTANDING_END|>"
            text_ids = tokenizer.encode(text, add_special_tokens=False)
            
            # Process audio
            waveform = torch.tensor(example["audio"]["array"]).float()

            if example["audio"]["sampling_rate"] != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, example["audio"]["sampling_rate"], sample_rate
                )

            with torch.no_grad():
                speech_codes = codec_model.encode_code(waveform.unsqueeze(0).cuda())[0, 0]

            speech_ids = (
                [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
                + [tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes.cpu().numpy()]
                + [tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
            )

            # Calculate available space
            MAX_TEXT_SPACE = max_length - len(speech_ids)
            if MAX_TEXT_SPACE < 0:
                print(f"Warning: Speech sequence too long ({len(speech_ids)} tokens) for sample {idx}, skipping...")
                continue

            # Truncate text to fit
            truncated_text = text_ids[:MAX_TEXT_SPACE]
            
            if debug and idx == 0:
                print(f"\nTruncated text tokens: {len(truncated_text)} (max available: {MAX_TEXT_SPACE})")

            # Build final sequence
            final_sequence = (
                truncated_text
                + speech_ids
                + [tokenizer.pad_token_id] * (max_length - len(truncated_text) - len(speech_ids))
            )[:max_length]
            all_sequences.append(final_sequence)

        if all_sequences:  # Only save if we have sequences
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio dataset for LLM training')
    
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Path to the dataset or HuggingFace dataset name')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed dataset')
    parser.add_argument('--tokenizer_name', type=str, 
                      default="HuggingFaceTB/SmolLM2-360M-Instruct",
                      help='Name or path of the tokenizer')
    parser.add_argument('--xcodec2_model_name', type=str,
                      default="HKUST-Audio/xcodec2",
                      help='Name or path of the XCodec2 model')
    parser.add_argument('--sample_rate', type=int,
                      default=16000,
                      help='Target sample rate for audio')
    parser.add_argument('--max_length', type=int,
                      default=2048,
                      help='Maximum sequence length')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode (process only 10 samples)')
    parser.add_argument('--subset_ratio', type=float,
                      default=1.0,
                      help='Ratio of dataset to process (0.0-1.0)')

    args = parser.parse_args()

    # Validate subset_ratio
    if not 0.0 < args.subset_ratio <= 1.0:
        raise ValueError("subset_ratio must be between 0.0 and 1.0")

    preprocess_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        xcodec2_model_name=args.xcodec2_model_name,
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        debug=args.debug,
        subset_ratio=args.subset_ratio
    )
