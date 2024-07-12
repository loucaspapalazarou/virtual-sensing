import subprocess
from tqdm import tqdm


def run_program_in_conda_env(env_name, program, script, iterations, args):
    """
    Runs a program from a specific conda environment.

    Parameters:
    env_name (str): The name of the conda environment.
    program (str): The program to run.
    script (str): The script or program path to run.
    iterations (int): Number of iterations to run the program.
    args (str): Additional arguments to pass to the program.
    """
    for _ in tqdm(range(iterations)):
        try:
            # Build the command to activate the conda environment and run the program
            command = f"conda run -n {env_name} {program} {script} {args}"

            # Execute the command
            result = subprocess.run(
                command, shell=True, check=True, capture_output=True, text=True
            )

            # Print the output and error (if any)
            print("Output:\n", result.stdout)
            if result.stderr:
                print("Error:\n", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running the program: {e}")


# Example usage
if __name__ == "__main__":
    iterations = 100
    num_envs = 12
    env_name = "rlgpu"
    program = "python"
    script = "/mnt/BigHD_1/loucas/IsaacGymEnvs/isaacgymenvs/train.py"
    args = f"task=FrankaCubePush headless=True checkpoint=/mnt/BigHD_1/loucas/IsaacGymEnvs/isaacgymenvs/runs/FrankaCubePush_09-10-42-48/nn/last_FrankaCubePush_ep_6500_rew_389.39374.pth test=True num_envs={num_envs}"
    run_program_in_conda_env(env_name, program, script, iterations, args)
