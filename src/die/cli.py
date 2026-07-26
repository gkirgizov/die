"""Command-line interface for DIE."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from die.registry import AgentId, DynamicsId, load_agent, make_agent, make_env

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help='DIE -- Distributed Intelligence Environment. '
         'Artificial Life simulations of learning cellular automata.',
)

SeedOpt = Annotated[int | None, typer.Option(help='Random seed for reproducible runs.')]
FieldOpt = Annotated[int, typer.Option(min=8, help='Field side length (square field).')]
RatioOpt = Annotated[float, typer.Option(min=0., max=1., help='Initial agent density.')]
FpsOpt = Annotated[int, typer.Option(min=1, help='Animation frames per second.')]
LogLevelOpt = Annotated[str, typer.Option(help='Python logging level.')]
NormalizeOpt = Annotated[bool, typer.Option(
    '--normalize/--no-normalize',
    help='Rescale unbounded channels to the display range instead of letting '
         'them saturate. Trades cross-channel color comparison for detail.')]


def _setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.WARNING))


def _frame_source(env, agent, normalize: bool, renders: bool):
    """Builds the frame source, or None when nothing will consume frames.

    Conditional so a headless run without recording never imports the view
    layer (and with it matplotlib).
    """
    if not renders:
        return None
    from die.view.render import FrameSource
    return FrameSource(env, agent, normalize=normalize)


def _build_observers(headless: bool, record: Path | None, fps: int) -> list:
    from die.sim import MetricsObserver, TqdmObserver

    observers = [TqdmObserver(), MetricsObserver()]
    if record is not None:
        from die.view.record import AnimationRecorder
        observers.append(AnimationRecorder(record, fps=fps))
    if not headless:
        from die.view.interactive import InteractiveViewObserver
        observers.append(InteractiveViewObserver())
    return observers


def _report(result) -> None:
    typer.echo(f'Finished after {result.num_steps} steps: '
               f'total_reward={result.total_reward:.3f}, '
               f'agents_alive={result.final_info.get("num_agents", "?")}'
               + (' [terminated]' if result.terminated else ''))


@app.command()
def run(agent: Annotated[AgentId, typer.Option(help='Agent behavior.')] = AgentId.physarum,
        dynamics: Annotated[DynamicsId, typer.Option(help='Environment dynamics preset.')]
            = DynamicsId.st_perlin,
        steps: Annotated[int, typer.Option(min=1)] = 1000,
        field_size: FieldOpt = 156,
        agent_ratio: RatioOpt = 0.1,
        seed: SeedOpt = None,
        headless: Annotated[bool, typer.Option('--headless/--view',
                                               help='Disable the live view.')] = False,
        record: Annotated[Path | None,
                          typer.Option(help='Record the run to a .gif/.mp4 file.')] = None,
        fps: FpsOpt = 10,
        normalize: NormalizeOpt = True,
        log_level: LogLevelOpt = 'WARNING'):
    """Run a scripted agent in the environment."""
    _setup_logging(log_level)
    from die.sim import Simulation

    env = make_env(dynamics, field_size, agent_ratio=agent_ratio)
    agent_obj = make_agent(agent, field_size, seed=seed)
    sim = Simulation(env, agent_obj, seed=seed,
                     observers=_build_observers(headless, record, fps),
                     frame_source=_frame_source(
                         env, agent_obj, normalize,
                         renders=not headless or record is not None))
    result = sim.run(steps)
    _report(result)
    if record:
        typer.echo(f'Animation saved to: {record}')


@app.command()
def record(agent: Annotated[AgentId, typer.Option(help='Agent behavior.')] = AgentId.physarum,
           dynamics: Annotated[DynamicsId, typer.Option(help='Environment dynamics preset.')]
               = DynamicsId.st_perlin,
           steps: Annotated[int, typer.Option(min=1)] = 150,
           field_size: FieldOpt = 196,
           agent_ratio: RatioOpt = 0.15,
           seed: SeedOpt = None,
           fps: FpsOpt = 10,
           normalize: NormalizeOpt = True,
           out: Annotated[Path, typer.Option(
               help='Output file (.gif/.mp4) or directory.')] = Path('samples')):
    """Record an animation of a run, headless (no display needed)."""
    _setup_logging('WARNING')
    from die.sim import Simulation
    from die.view.record import AnimationRecorder

    env = make_env(dynamics, field_size, agent_ratio=agent_ratio)
    agent_obj = make_agent(agent, field_size, seed=seed)
    if out.suffix == '':
        out = out / f'{type(agent_obj).__name__}.gif'

    from die.sim import TqdmObserver
    sim = Simulation(env, agent_obj, seed=seed,
                     observers=[TqdmObserver(), AnimationRecorder(out, fps=fps)],
                     frame_source=_frame_source(env, agent_obj, normalize,
                                                renders=True))
    result = sim.run(steps)
    _report(result)
    typer.echo(f'Animation saved to: {out}')


@app.command()
def train(dynamics: Annotated[DynamicsId, typer.Option(help='Environment dynamics preset.')]
              = DynamicsId.st_perlin_wide,
          generations: Annotated[int, typer.Option(min=1)] = 100,
          episode_steps: Annotated[int, typer.Option(min=1,
              help='Env steps per fitness evaluation.')] = 50,
          popsize: Annotated[int, typer.Option(min=2)] = 10,
          field_size: FieldOpt = 96,
          agent_ratio: RatioOpt = 0.1,
          seed: SeedOpt = None,
          algorithm: Annotated[str, typer.Option(help="Search algorithm: 'pgpe' or 'cmaes'.")]
              = 'pgpe',
          out: Annotated[Path, typer.Option(help='Output directory for checkpoints, '
                                                 'rollouts and (by default) mlruns.')]
              = Path('runs/latest'),
          mlflow_experiment: Annotated[str, typer.Option(help='MLflow experiment name.')]
              = 'die',
          mlflow_uri: Annotated[str | None, typer.Option(
              help='MLflow tracking URI '
                   '(default: MLFLOW_TRACKING_URI or sqlite db in <out>).')]
              = None,
          no_mlflow: Annotated[bool, typer.Option('--no-mlflow',
              help='Disable MLflow tracking.')] = False,
          checkpoint_every: Annotated[int, typer.Option(min=0)] = 25,
          record_every: Annotated[int, typer.Option(min=0,
              help='Record a rollout animation of the best agent every N generations '
                   '(0 disables).')] = 25,
          record_steps: Annotated[int, typer.Option(min=1)] = 100,
          eval_seed_mode: Annotated[str, typer.Option(
              help="'per_generation' (default) or 'fixed' env init per evaluation.")]
              = 'per_generation',
          log_level: LogLevelOpt = 'WARNING'):
    """Train a neural cellular automata agent with evolution strategies."""
    _setup_logging(log_level)
    import os

    try:
        from die.train.evotorch_backend import EvotorchBackend
    except ImportError as exc:
        typer.echo(f'Error: {exc}', err=True)
        raise typer.Exit(code=1) from exc
    from die.train import EvolutionTrainer, MlflowTrainLogger, StdoutTrainLogger, TrainConfig

    env = make_env(dynamics, field_size, agent_ratio=agent_ratio)
    agent_obj = make_agent(AgentId.neural, field_size, seed=seed)

    config = TrainConfig(generations=generations,
                         episode_steps=episode_steps,
                         popsize=popsize,
                         seed=seed,
                         eval_seed_mode=eval_seed_mode,
                         checkpoint_every=checkpoint_every,
                         record_every=record_every,
                         record_steps=record_steps,
                         out_dir=out)
    loggers: list = [StdoutTrainLogger()]
    if not no_mlflow:
        out.mkdir(parents=True, exist_ok=True)
        uri = mlflow_uri or os.environ.get('MLFLOW_TRACKING_URI') \
            or f'sqlite:///{(out / "mlflow.db").absolute()}'
        loggers.append(MlflowTrainLogger(
            experiment=mlflow_experiment, tracking_uri=uri,
            artifact_location=str((out / 'mlartifacts').absolute())))
        typer.echo(f'MLflow tracking: {uri}\n'
                   f'  browse with: mlflow ui --backend-store-uri "{uri}"')

    trainer = EvolutionTrainer(
        env=env, agent=agent_obj,
        backend=EvotorchBackend(algorithm=algorithm),
        config=config, loggers=loggers,
        run_context={'dynamics': str(dynamics), 'field_size': field_size,
                     'agent_ratio': agent_ratio, 'agent': str(AgentId.neural)})
    trainer.train()
    typer.echo(f'Best agent checkpoint: {out / "checkpoints" / "best.pt"}')
    typer.echo(f'Replay it with: die replay {out / "checkpoints" / "best.pt"} '
               f'--dynamics {dynamics} --field-size {field_size}')


@app.command()
def replay(model: Annotated[Path, typer.Argument(help='Trained agent checkpoint (.pt).',
                                                 exists=True, readable=True)],
           dynamics: Annotated[DynamicsId, typer.Option(help='Environment dynamics preset.')]
               = DynamicsId.st_perlin_wide,
           steps: Annotated[int, typer.Option(min=1)] = 1000,
           field_size: FieldOpt = 96,
           agent_ratio: RatioOpt = 0.1,
           seed: SeedOpt = None,
           headless: Annotated[bool, typer.Option('--headless/--view',
                                                  help='Disable the live view.')] = False,
           record: Annotated[Path | None,
                             typer.Option(help='Record the run to a .gif/.mp4 file.')] = None,
           fps: FpsOpt = 10,
           normalize: NormalizeOpt = True,
           log_level: LogLevelOpt = 'WARNING'):
    """Replay a trained (saved) agent in the environment."""
    _setup_logging(log_level)
    from die.sim import Simulation

    env = make_env(dynamics, field_size, agent_ratio=agent_ratio)
    agent_obj = load_agent(model)
    sim = Simulation(env, agent_obj, seed=seed,
                     observers=_build_observers(headless, record, fps),
                     frame_source=_frame_source(
                         env, agent_obj, normalize,
                         renders=not headless or record is not None))
    result = sim.run(steps)
    _report(result)


if __name__ == '__main__':
    app()
