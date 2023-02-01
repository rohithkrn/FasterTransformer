import subprocess


def execute_command(command: str):
    subprocess.check_call(command, shell=True)