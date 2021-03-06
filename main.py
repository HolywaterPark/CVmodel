import typer

from train import app as train_app

app = typer.Typer()

app.add_typer(train_app, name='trainer')

if __name__ == "__main__":
    app()
