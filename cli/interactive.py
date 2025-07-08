import typer
from rich.console import Console
from rich.table import Table
from model.predict import predict_next_token

console = Console()
cli = typer.Typer(name="interactive")

@cli.command()
def start(top_k: int = 5):
    """
    Start an interactive autocomplete shell.
    """
    console.print("[bold green]Terminal Autocomplete[/bold green] (type 'exit' to quit)\n")

    while True:
        try:
            prompt = console.input("[bold blue]>>>[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]👋 Exiting...[/bold red]")
            break

        if prompt.strip().lower() in {"exit", "quit"}:
            console.print("[bold red]👋 Exiting...[/bold red]")
            break

        predictions = predict_next_token(prompt, top_k=top_k)

        table = Table(title=f"Top {top_k} Suggestions for: [cyan]'{prompt}'[/cyan]")
        table.add_column("Token", justify="left", style="magenta", no_wrap=True)
        table.add_column("Probability", justify="right", style="green")

        for token, prob in predictions:
            table.add_row(token, f"{prob:.4f}")

        console.print(table)
