import io
import os
import tarfile
from pathlib import Path
from time import sleep

import docker

client = docker.from_env()


def copy_to_container(container, folder, dest):
    try:
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(folder, arcname=dest)
        tar_stream.seek(0)
        container.put_archive("/", tar_stream)
        container.exec_run(["sudo", "chown", "-R", "ubuntu", dest])
    except Exception as e:
        print(f"Error: {e}")


def copy_from_container(container, folder, dest):
    try:
        stream, _ = container.get_archive(folder)
        tar_stream = io.BytesIO()
        for chunk in stream:
            tar_stream.write(chunk)
        tar_stream.seek(0)
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            for member in tar.getmembers():
                f = tar.extractfile(member)
                if f:
                    with open(dest, "wb") as out_file:
                        out_file.write(f.read())
    except Exception as e:
        print(f"Error: {e}")


class LCBEnv:
    def __init__(self, **docker_kwargs):
        self.client = docker.from_env()
        self.start_container(**docker_kwargs)

    def start_container(self, **docker_kwargs):
        self.container = self.client.containers.run(
            "moogician/livecodebench",
            "zsh",
            detach=True,
            tty=True,
            stdout=True,
            stderr=True,
            network_mode="host",
            **docker_kwargs,
        )
        try:
            while self.container.status != "running":
                sleep(1)
                self.container.reload()
        except Exception as e:
            raise RuntimeError("Container start error", e)

    def __call__(self, command):
        try:
            exit_code, output = self.container.exec_run(["zsh", "-c", command])
            output = output.decode("utf-8")
            if exit_code != 0:
                raise RuntimeError(
                    f"Experiment error with code {exit_code}", output
                )
        except Exception as e:
            raise RuntimeError("Experiment error", e)
        return output

    def eval(self, generated_dir: Path, nproc=16) -> str:
        copy_to_container(
            self.container,
            folder=str(generated_dir.absolute() / "all_result.json"),
            dest="/home/ubuntu/livecodebench/all_result.json",
        )
        command = (
            "source ~/.zshrc && source .venv/bin/activate && "
            "python -m lcb_runner.runner.main --scenario codegeneration "
            "--multiprocess 32 --eval_file all_result.json --evaluate"
        )
        return self(command)

    def collect(self, after_root):
        copy_from_container(
            self.container,
            "/home/ubuntu/livecodebench/all_result_eval.json",
            after_root / "result.json",
        )
        copy_from_container(
            self.container,
            "/home/ubuntu/livecodebench/all_result_eval_all.json",
            after_root / "all_result.json",
        )

    def shutdown(self):
        try:
            self.container.stop()
            self.container.remove()
        except Exception as e:
            raise RuntimeError("Container stop error", repr(e))
