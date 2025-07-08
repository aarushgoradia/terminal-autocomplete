import typer
from cli import interactive

app = typer.Typer()
app.add_typer(interactive.cli, name="interactive")

if __name__ == "__main__":
    app()
