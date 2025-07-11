from data.data_module import ParquetStreamDataset
from models.transformer import Decoder, DecoderLightning
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from argparse import ArgumentParser
from types import SimpleNamespace
import json


def train(config):
    # Load tokenizer
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    # Get dataset and dataloader ready
    train_dataset = ParquetStreamDataset(
        config.train_data, 
        column=config.text_column, 
        batch_size=config.dataset_batch_size, 
        tokenizer=tokenizer,
        context_window=config.context_window)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers, 
        prefetch_factor=config.prefetch_factor, 
        pin_memory=config.pin_memory
    )

    # Get the model ready
    decoder = Decoder(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        vocab_size=tokenizer.get_vocab_size(),
        padding_idx=tokenizer.token_to_id('[PAD]'),
        ff_factor=config.ff_factor,
        dropout=config.dropout,
        max_len=config.max_len
    )

    model = DecoderLightning(
        model=decoder, 
        label_smoothing=config.label_smoothing, 
        learning_rate=config.learning_rate, 
        warmup_steps=config.warmup_steps, 
        betas=[config.beta1, config.beta2], 
        eps=config.eps
    )

    # Get the callbacks ready
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.checkpoint_path,
        monitor=config.checkpoint_monitor,
        save_top_k=config.save_top,
        mode=config.mode,
        every_n_train_steps=config.every_n_train_steps
    )

    logger = TensorBoardLogger(
        save_dir=config.logger_path,
        name='logs'
    )

    # Get the trainer ready
    trainer = Trainer(
        devices=config.devices, 
        max_epochs=config.epochs,
        precision=config.precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        callbacks=checkpoint_callback,
        log_every_n_steps=10,
        logger=logger
    )

    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        ckpt_path=config.ckpt_path
    )


if __name__ == '__main__':
    parser = ArgumentParser('No description')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        file = json.load(file)
        config = SimpleNamespace(file)
    
    train(config=config)