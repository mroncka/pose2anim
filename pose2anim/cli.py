"""CLI entry point for pose2anim."""

import logging

import click
from rich.console import Console
from rich.logging import RichHandler

from pose2anim.pipeline import Pose2AnimPipeline

console = Console()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool):
    """pose2anim - Video to 3D Animation Pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@main.command()
@click.option("--input", "-i", required=True, help="Input video file path")
@click.option("--output", "-o", required=True, help="Output animation file path")
@click.option("--config", "-c", default=None, help="Config YAML file path")
def process(input: str, output: str, config: str):
    """Process a video file to animation."""
    console.print(f"[bold green]Processing:[/] {input} → {output}")

    pipeline = Pose2AnimPipeline(config_path=config)
    result = pipeline.process_video(input, output)

    console.print(f"[bold green]Done![/] Animation saved to: {result}")


@main.command()
@click.option("--camera", default=0, help="Camera device index")
@click.option("--config", "-c", default=None, help="Config YAML file path")
def live(camera: int, config: str):
    """Real-time pose estimation from webcam."""
    console.print(f"[bold green]Starting live capture[/] (camera {camera})")
    console.print("Press [bold]Q[/] to quit")

    pipeline = Pose2AnimPipeline(config_path=config)
    pipeline.process_live(camera_id=camera)


if __name__ == "__main__":
    main()
