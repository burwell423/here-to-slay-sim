from .models import Policy


def run_game(*args, **kwargs):
    from .game import run_game as _run_game

    return _run_game(*args, **kwargs)


__all__ = ["run_game", "Policy"]
