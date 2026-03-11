import asyncio
import logging
from typing import Optional

import cv2
import gymnasium as gym
import ipywidgets as widgets
from IPython.display import display

from src.environments import clara
from src.render_3d.beam_server.beam_visualization_wrapper import (
    BeamVisualizationWrapper,
)


async def run_simulation_async(
    env: BeamVisualizationWrapper,
    simulation_task: asyncio.Task,
    logger: Optional[logging.Logger] = None,
):
    """Run the simulation loop, stepping the environment with control actions."""
    print("\n--- Starting simulation loop ---")

    step_count = 0
    done = False
    wspromise = env.start_websocket()

    # Reset the environment
    observation, _ = env.reset()

    while not done:
        # Check if we have a new control action from WebSocket
        if env.control_action is not None:
            # print(f"New action received: {env.control_action}")
            env.last_action = env.control_action  # Update last_action with new action
            env.control_action = None  # Clear after use

            # Step through the environment with the last action
            # Note: last_action persists and is reused if no new action is received
            observation, reward, terminated, truncated, info = env.step(env.last_action)

            # Update step count
            step_count += 1

            if logger is not None:
                logger.debug(
                    "Step %d: Action = %s, Reward = %s, Observation = %s",
                    step_count,
                    env.last_action,
                    reward,
                    observation,
                )

            done = truncated  # or terminated
            # Render and broadcast data to clients
            # Calls BeamVisualizationWrapper.render(),
            # which delegates to WebSocketWrapper.render()
            await env.render()

        if info["stop_simulation"]:  # Stop the simulation if truncated is True
            print("Truncated flag is True, stopping simulation...")
            simulation_task.cancel()
            break

    env.close()
    await wspromise  # Ensure WebSocket server is properly closed
    print("Simulation completed.")


async def visualization_main(logger: Optional[logging.Logger] = None):
    """Main entry point to set up the environment and start the simulation."""
    # Initialize the environment and wrap it with BeamVisualizationWrapper
    env = clara.TransverseTuning(
        backend="cheetah",
        action_mode="direct",
        magnet_init_mode=None,
        render_mode="rgb_array",  # "human",
        backend_args={"generate_screen_images": True},
    )
    env = BeamVisualizationWrapper(env)
    env.reset()

    if not env.connected:
        print("Waiting for WebSocket client to connect...")
        # await asyncio.sleep(5.0)  # Small delay to prevent CPU overload

    # Create the background simulation task
    simulation_task = asyncio.create_task(env.start_websocket())

    # Return the task to allow cancellation or monitoring if needed
    return simulation_task


def restart_manual_tuning(env: gym.Env):
    """Restart the manual tuning process and display in the jupyter notebook cell."""

    assert isinstance(env.env, clara.TransverseTuning)

    observation, info = env.reset()

    # magnet_names = info["magnet_names"]
    magnet_names = [
        "SO4-HCOR-01",
        "SO4-VCOR-01",
        "SO4-QUAD-01",
        "SO4-QUAD-02",
        "SO4-QUAD-03",
    ]
    magnet_mins = env.observation_space["magnets"].low
    magnet_maxs = env.observation_space["magnets"].high
    current_magnet_settings = observation["magnets"]

    magnet_mins[0] = magnet_mins[0] * 1e3
    magnet_mins[1] = magnet_mins[1] * 1e3
    magnet_maxs[0] = magnet_maxs[0] * 1e3
    magnet_maxs[1] = magnet_maxs[1] * 1e3
    current_magnet_settings[0] = current_magnet_settings[0] * 1e3
    current_magnet_settings[1] = current_magnet_settings[1] * 1e3

    image = env.render()

    magnet_widgets = [
        widgets.BoundedFloatText(
            value=setting,
            description=name,
            min=min_limit,
            max=max_limit,
            style={"description_width": "150px"},
        )
        for name, min_limit, max_limit, setting in zip(
            magnet_names, magnet_mins, magnet_maxs, current_magnet_settings
        )
    ]
    for w in magnet_widgets:
        w.observe(lambda _: update_screen_image(), names="value")

    done_button = widgets.Button(description="Done!")
    done_button.on_click(lambda _: on_done_button_clicked())

    # Output widget for displaying the final figure
    output_fig = widgets.Output()

    image_flipped = cv2.flip(image, 0)  # Flip vertically (0 for vertical flip)
    bgr = cv2.cvtColor(image_flipped, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", bgr)
    screen_image_widget = widgets.Image(
        value=buffer, format="png", width=500, height=500
    )

    def update_screen_image():
        action = [widget.value for widget in magnet_widgets]
        action[0] = action[0] * 1e-3
        action[1] = action[1] * 1e-3
        observation, _, _, _, _ = env.step(action)

        for i, (w, setting) in enumerate(zip(magnet_widgets, observation["magnets"])):
            if i == 0 or i == 1:
                w.value = setting * 1e3
            else:
                w.value = setting

        image = env.render()
        image_flipped = cv2.flip(image, 0)  # Flip vertically
        bgr = cv2.cvtColor(image_flipped, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".png", bgr)
        screen_image_widget.value = buffer

    def on_done_button_clicked():
        for w in magnet_widgets:
            w.close()

        done_button.close()

        screen_image_widget.close()
        # Display the figure inside the Output widget
        with output_fig:
            output_fig.clear_output(wait=True)
            fig = env.generate_episode_plot()
            display(fig)
        env.reset()

    display(*magnet_widgets, screen_image_widget, done_button, output_fig)
