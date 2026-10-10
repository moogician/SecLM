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
            tar.add(folder, arcname=".")
        tar_stream.seek(0)
        container.put_archive(dest, tar_stream)
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
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            tar.extractall(dest)
    except Exception as e:
        print(f"Error: {e}")


class CWEvalEnv:
    def __init__(self, **docker_kwargs):
        self.client = docker.from_env()
        self.start_container(**docker_kwargs)

    def start_container(self, **docker_kwargs):
        self.container = self.client.containers.run(
            "co1lin/cweval",
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

        exit_code, result = self.container.exec_run(
            [
                "zsh",
                "-c",
                r"sed -i '92s/not _unsafe/(not _unsafe) and (not cwe_400_0_test.py)/' cweval/run_tests.py",
            ]
        )
        assert exit_code == 0, result
        exit_code, result = self.container.exec_run(
            [
                "zsh",
                "-c",
                "sed -i '186 a\\\n    if \"</think>\" in msg:\\\n        msg = msg.split(\"</think>\")[-1]' cweval/commons.py",
            ]
        )
        assert exit_code == 0, result

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
        self.container.exec_run(["mkdir", "-p", f"{generated_dir.name}"])
        copy_to_container(
            self.container,
            folder=str(generated_dir.absolute()),
            dest=f"/home/ubuntu/CWEval/{generated_dir.name}",
        )
        command = (
            "source ~/.zshrc && source .env && "
            + f"python cweval/evaluate.py pipeline --eval_path {generated_dir.name} --num_proc {nproc} --docker False"
        )
        return self(command)

    def collect(self, generated_dir, after_root):
        copy_from_container(
            self.container,
            f"/home/ubuntu/CWEval/{generated_dir.name}",
            after_root,
        )

    def shutdown(self):
        try:
            self.container.stop()
            self.container.remove()
        except Exception as e:
            raise RuntimeError("Container stop error", repr(e))
