"""
Unified CLI for LLM Training workflows.
"""
import typer
from omegaconf import OmegaConf
from llm_training.data import prepare_data
from llm_training.training import train_model
from llm_training.evaluation import evaluate_model
from llm_training.utils import setup_logging

app = typer.Typer(help="Unified CLI for LLM Training workflows.")

@app.command()
def data_prep(config: str = typer.Option(..., help="Path to YAML config.")):
    """Prepare data for LLM training."""
    setup_logging()
    cfg = OmegaConf.load(config)
    prepare_data(
        dataset_name=cfg.dataset_name,
        tokenizer_name=cfg.tokenizer_name,
        sample_percentage=cfg.sample_percentage,
        split_ratio=cfg.split_ratio,
        output_dir=cfg.output_dir,
    )
    typer.echo(f"Data prepared and saved to {cfg.output_dir}")

@app.command()
def train(config: str = typer.Option(..., help="Path to YAML config.")):
    """Fine-tune an LLM."""
    setup_logging()
    cfg = OmegaConf.load(config)
    train_model(OmegaConf.to_container(cfg, resolve=True))
    typer.echo("Training complete.")

@app.command()
def eval(config: str = typer.Option(..., help="Path to YAML config.")):
    """Evaluate an LLM."""
    setup_logging()
    cfg = OmegaConf.load(config)
    evaluate_model(OmegaConf.to_container(cfg, resolve=True))
    typer.echo("Evaluation complete.") 