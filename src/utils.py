import inspect
import logging
import os
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import wandb
import yaml
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)

plt.style.use(["science", "nature", "no-latex"])

DEFAULT_REWARD_IMPORTS = [
    "import math",
    "import numpy as np",
    "from src.environments.clara import TransverseTuning",
]


def evaluate_mae(observations) -> tuple[plt.Figure, plt.Axes]:
    maes = np.array(
        [np.mean(np.abs(obs["beam"] - obs["target"])) for obs in observations]
    )

    fig, ax = plt.subplots()
    ax.semilogy(maes * 1000)
    ax.set_ylabel("Mean Absolute Error (mm)")
    ax.set_xlabel("Step")

    print(f"Minimum MAE:                {min(maes) * 1000:.3f}  mm")
    print(f"Sum of MAE over all steps:  {np.sum(maes) * 1000:.3f} mm")

    return fig, ax


def load_config(path: str) -> dict:
    """
    Load a training setup config file to a config dictionary. The config file must be a
    `.yaml` file. The `path` argument to this function should be given without the file
    extension.
    """
    with open(f"{path}.yaml", "r") as f:
        data = yaml.load(f.read(), Loader=yaml.Loader)
    return data


def plot_beam_history(ax, observations, before_reset=None):
    mu_x = np.array([obs["beam"][0] for obs in observations])
    sigma_x = np.array([obs["beam"][1] for obs in observations])
    mu_y = np.array([obs["beam"][2] for obs in observations])
    sigma_y = np.array([obs["beam"][3] for obs in observations])

    if before_reset is not None:
        mu_x = np.insert(mu_x, 0, before_reset[0])
        sigma_x = np.insert(sigma_x, 0, before_reset[1])
        mu_y = np.insert(mu_y, 0, before_reset[2])
        sigma_y = np.insert(sigma_y, 0, before_reset[3])

    target_beam = observations[0]["target"]

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Beam Parameters")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("(mm)")
    ax.plot(steps, mu_x * 1e3, label=r"$\mu_x$", c="tab:blue")
    ax.plot(steps, [target_beam[0] * 1e3] * len(steps), ls="--", c="tab:blue")
    ax.plot(steps, sigma_x * 1e3, label=r"$\sigma_x$", c="tab:orange")
    ax.plot(steps, [target_beam[1] * 1e3] * len(steps), ls="--", c="tab:orange")
    ax.plot(steps, mu_y * 1e3, label=r"$\mu_y$", c="tab:green")
    ax.plot(steps, [target_beam[2] * 1e3] * len(steps), ls="--", c="tab:green")
    ax.plot(steps, sigma_y * 1e3, label=r"$\sigma_y$", c="tab:red")
    ax.plot(steps, [target_beam[3] * 1e3] * len(steps), ls="--", c="tab:red")
    ax.legend()
    ax.grid(True)


def plot_screen_image(ax, img, screen_resolution, pixel_size, title="Beam Image"):
    screen_size = screen_resolution * pixel_size

    ax.set_title(title)
    ax.set_xlabel("(mm)")
    ax.set_ylabel("(mm)")
    ax.imshow(
        img,
        vmin=0,
        aspect="equal",
        interpolation="none",
        extent=(
            -screen_size[0] / 2 * 1e3,
            screen_size[0] / 2 * 1e3,
            -screen_size[1] / 2 * 1e3,
            screen_size[1] / 2 * 1e3,
        ),
    )


def plot_quadrupole_history(ax, observations, before_reset=None):
    areamqzm1 = [obs["magnets"][2] for obs in observations]
    areamqzm2 = [obs["magnets"][3] for obs in observations]
    areamqzm3 = [obs["magnets"][4] for obs in observations]

    if before_reset is not None:
        areamqzm1 = [before_reset[0]] + areamqzm1
        areamqzm2 = [before_reset[1]] + areamqzm2
        areamqzm3 = [before_reset[3]] + areamqzm3

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Quadrupoles")
    ax.set_xlim([start, len(observations) + 1])
    ax.set_xlabel("Step")
    ax.set_ylabel("Strength (1/m^2)")
    ax.plot(steps, areamqzm1, label="AREAMQZM1")
    ax.plot(steps, areamqzm2, label="AREAMQZM2")
    ax.plot(steps, areamqzm3, label="AREAMQZM3")
    ax.legend()
    ax.grid(True)


def plot_steerer_history(ax, observations, before_reset=None):
    areamcvm1 = np.array([obs["magnets"][0] for obs in observations])
    areamchm2 = np.array([obs["magnets"][1] for obs in observations])

    if before_reset is not None:
        areamcvm1 = np.insert(areamcvm1, 0, before_reset[2])
        areamchm2 = np.insert(areamchm2, 0, before_reset[4])

    start = 0 if before_reset is None else -1
    steps = np.arange(start, len(observations))

    ax.set_title("Steerers")
    ax.set_xlabel("Step")
    ax.set_ylabel("Kick (mrad)")
    ax.set_xlim([start, len(observations) + 1])
    ax.plot(steps, areamcvm1 * 1e3, label="AREAMCVM1")
    ax.plot(steps, areamchm2 * 1e3, label="AREAMCHM2")
    ax.legend()
    ax.grid(True)


def remove_if_exists(path):
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def save_config(data: dict, path: str) -> None:
    """
    Save a training setup config to a `.yaml` file. The `path` argument to this function
    should be given without the file extension.
    """
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


def save_custom_reward_env(
    tmp_file: str | Path, algorithm: Literal["ppo", "sac", "td3"], path: str
) -> None:
    """Save custom reward environment class defined in challenge notebook
        to a Python file.

    :param custom_env: The custom reward environment class to save.
    :param algorithm: The RL algorithm used (affects filename).
    :param path: The directory path where to save the custom environment file.
    """

    save_path = Path(path) / "custom_reward_environment.py"

    # Check if temporary file exists (created by %%writefile magic)
    tmp_file = Path(tmp_file)
    if tmp_file.exists():
        # Copy and prepend imports
        imports = """\"\"\"
Custom reward environment for RL training.
Auto-generated from training notebook.
\"\"\"
import math
import numpy as np
from src.environments.clara import TransverseTuning


"""
        content = tmp_file.read_text()
        save_path.write_text(imports + content)
        tmp_file.unlink()  # Clean up temp file
        print(f"Saved custom environment to {save_path}")
    else:
        raise FileNotFoundError(
            f"Could not find tmp/temp_custom_env_{algorithm}.py. "
            "Make sure to add %%writefile tmp/temp_custom_env_{algorithm}.py "
            "at the start of the cell where CustomRewardEnvironment is defined."
        )


def save_reward_source(
    reward_cls,
    path: str | Path,
    extra_imports: list[str] | None = None,
    filename: str = "custom_reward_environment.py",
) -> Path | None:
    """
    Save the source code of a reward class with its imports to a file.

    Tries `inspect.getsource()` first, falls back to searching IPython's
    input history if that fails (e.g. when running in a Jupyter notebook).

    If the source cannot be retrieved, a warning is logged and None is returned.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename

    all_imports = DEFAULT_REWARD_IMPORTS + (extra_imports or [])
    header = "\n".join(all_imports) + "\n\n\n"

    # Try inspect.getsource first
    try:
        source = textwrap.dedent(inspect.getsource(reward_cls))
    except OSError:
        # Fallback: search IPython input history for the cell that defined the class
        try:
            source = _get_source_from_ipython_history(reward_cls.__name__)
        except Exception as e:
            logger.warning(
                f"Could not save reward source for {reward_cls.__name__}: {e}\n"
                "Training will continue, but the reward source will not be saved. "
                "Please keep track of your reward implementation manually."
            )
            return None

    filepath.write_text(header + source)
    print(f"Saved custom environment to {filepath}")
    return filepath


def _get_source_from_ipython_history(class_name: str) -> str:
    """
    Search IPython's input history for the cell that defined a class.
    Returns only the class definition (and any decorators), not the full cell.
    """
    try:
        import IPython

        shell = IPython.get_ipython()
        if shell is None:
            raise OSError(f"Cannot retrieve source for {class_name}")

        # Search history in reverse to find the most recent definition
        for cell_source in reversed(shell.user_ns["In"]):
            if f"class {class_name}" in cell_source:
                # Extract just the class definition from the cell
                lines = cell_source.split("\n")
                class_lines = []
                recording = False

                for line in lines:
                    stripped = line.lstrip()
                    if stripped.startswith(f"class {class_name}"):
                        recording = True
                        class_lines.append(line)
                    elif recording:
                        # Stop if we hit a non-indented, non-empty line
                        # (i.e. another top-level statement)
                        if (
                            stripped
                            and not line[0].isspace()
                            and not stripped.startswith("#")
                        ):
                            break
                        class_lines.append(line)

                if class_lines:
                    return textwrap.dedent("\n".join(class_lines)) + "\n"

        raise OSError(f"Could not find class {class_name} in IPython history")

    except (ImportError, KeyError) as e:
        raise OSError(
            f"Cannot retrieve source for {class_name}. "
            "Neither inspect.getsource() nor IPython history are available."
        ) from e


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix="rl_model",
        save_env=False,
        env_name_prefix="vec_normalize",
        save_replay_buffer=False,
        replay_buffer_name_prefix="replay_buffer",
        delete_old_replay_buffers=True,
        verbose=0,
    ):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_env = save_env
        self.env_name_prefix = env_name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.replay_buffer_name_prefix = replay_buffer_name_prefix
        self.delete_old_replay_buffers = delete_old_replay_buffers

    def _init_callback(self):
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Save model
            path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")

            # Save env (VecNormalize wrapper)
            if self.save_env:
                path = os.path.join(
                    self.save_path,
                    f"{self.env_name_prefix}_{self.num_timesteps}_steps.pkl",
                )
                self.training_env.save(path)
                if self.verbose > 1:
                    print(f"Saving environment to {path[:-4]}")

            # Save replay buffer
            if self.save_replay_buffer:
                path = os.path.join(
                    self.save_path,
                    f"{self.replay_buffer_name_prefix}_{self.num_timesteps}_steps",
                )
                self.model.save_replay_buffer(path)
                if self.verbose > 1:
                    print(f"Saving replay buffer to {path}")

                if self.delete_old_replay_buffers and hasattr(self, "last_saved_path"):
                    remove_if_exists(self.last_saved_path + ".pkl")
                    if self.verbose > 1:
                        print(f"Removing old replay buffer at {self.last_saved_path}")

                self.last_saved_path = path

        return True


class SLURMRescheduleCallback(BaseCallback):
    def __init__(self, reserved_time, safety=timedelta(minutes=1), verbose=0):
        super().__init__(verbose)
        self.allowed_time = reserved_time - safety
        self.t_start = datetime.now()
        self.t_last = self.t_start

    def _on_step(self):
        t_now = datetime.now()
        passed_time = t_now - self.t_start
        dt = t_now - self.t_last
        self.t_last = t_now
        if passed_time + dt > self.allowed_time:
            os.system(
                "sbatch"
                f" --export=ALL,WANDB_RESUME=allow,WANDB_RUN_ID={wandb.run.id} td3.sh"
            )
            if self.verbose > 1:
                print("Scheduling new batch job to continue training")
            return False
        else:
            if self.verbose > 1:
                print(
                    f"Continue running with this SLURM job (passed={passed_time} /"
                    f" allowed={self.allowed_time} / dt={dt})"
                )
            return True


def log_challenge_results(
    study_best: Any,
    study_final: Any,
    num_episode_plots: int = 3,
) -> tuple[float, str]:
    """
    Evaluate both studies, log scores and episode summary plots to the active
    wandb run, and return the challenge score and which model was best.

    :param study_best: Study from the best-evaluated model checkpoint.
    :param study_final: Study from the final model checkpoint.
    :param num_episode_plots: Number of episode summaries to log as images.
    :return: Tuple of (challenge_score, best_model_is, best_eval_model_score,
        final_model_score) where best_model_is is ``"best_evaluated"`` or
        ``"final"``.
    """
    print("Calculating score for best model from EvalCallback...")
    best_eval_model_score = study_best.evaluate_challenge()
    print("-----------------------------------------------------")
    print("Calculating score final model...")
    final_model_score = study_final.evaluate_challenge()
    challenge_score = min(best_eval_model_score, final_model_score)

    best_model_is = (
        "best_evaluated" if best_eval_model_score <= final_model_score else "final"
    )

    wandb.run.log(
        {
            "best_eval_model_score": best_eval_model_score,
            "final_model_score": final_model_score,
            "challenge_score": challenge_score,
        }
    )

    return challenge_score, best_model_is, best_eval_model_score, final_model_score


def deep_equal(obj1: Any, obj2: Any) -> bool:
    """Recursively checks if two objects are equal."""
    if type(obj1) is not type(obj2):
        return False

    if isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(deep_equal(obj1[key], obj2[key]) for key in obj1)

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(deep_equal(item1, item2) for item1, item2 in zip(obj1, obj2))

    if isinstance(obj1, np.ndarray):
        return np.array_equal(obj1, obj2)

    return obj1 == obj2
