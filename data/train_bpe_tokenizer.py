import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

def train_bpe_tokenizer(input_file, vocab_size, output_file):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Special tokens
    SOS = "[SOS]"
    EOS = "[EOS]"
    SEP = "[SEP]"
    special_tokens = ["[PAD]", "[UNK]", SOS, SEP, EOS, "[MASK]"]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        special_tokens=special_tokens
    )

    # Train
    tokenizer.train([input_file], trainer)

    # Set special tokens
    tokenizer.add_special_tokens(special_tokens)

    # Setup post-processor for CLS and SEP
    SOS_token_id = tokenizer.token_to_id(SOS)
    EOS_token_id = tokenizer.token_to_id(EOS)
    SEP_token_id = tokenizer.token_to_id(SEP)

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{SOS} $A {EOS}",
        pair=f"{SOS} $A {SEP} $B:1 {EOS}:1",
        special_tokens=[
            (SOS, SOS_token_id),
            (SEP, SEP_token_id),
            (EOS, EOS_token_id),
        ],
    )

    # Enable padding
    pad_token_id = tokenizer.token_to_id("[PAD]")
    tokenizer.enable_padding(
        pad_id=pad_token_id,
        pad_token="[PAD]"
    )

    # Save
    tokenizer.save(output_file)
    print(f"Tokenizer saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and save as JSON.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the trained tokenizer JSON.")

    args = parser.parse_args()
    train_bpe_tokenizer(args.input_file, args.vocab_size, args.output_file)