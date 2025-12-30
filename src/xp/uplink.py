"""Tools related to running experimentes remotely.

Requires rsync and ssh access to the server.
"""

import os
import subprocess
from contextlib import contextmanager
from pathlib import Path


class Uplink:
    """Multiplexed connection to `host` via ssh."""

    def __init__(self, host, progbar=False, dry=False, use_M=True):
        self.host = host
        self.progbar = progbar
        self.dry = dry
        self.use_M = use_M

        if os.name == "nt":  # Windows
            control_path = "%USERPROFILE%\\.ssh\\%r@%h:%p.socket"
        else:  # Unix-like (Linux, macOS, etc.)
            control_path = "~/.ssh/%r@%h:%p.socket"

        self.ssh_M = " ".join(
            [
                "ssh",
                "-o ControlMaster=auto",
                f"-o ControlPath={control_path}",
                "-o ControlPersist=1m",
            ]
        )

    def __repr__(self):
        return f"Uplink(host='{self.host}', progbar={self.progbar}, dry={self.dry}, use_M={self.use_M})"

    def cmd(self, cmd: str, login_shell=True, **kwargs):
        if isinstance(cmd, list):
            cmd = " ".join([str(x) for x in cmd])
        if login_shell:
            # sources ~/.bash_profile or ~/.profile, which may or not include ~/.bashrc
            cmd = f"bash -l -c '{cmd}'"

        kwargs = {**{"check": True, "text": True, "capture_output": True}, **kwargs}
        try:
            return subprocess.run([*self.ssh_M.split(), self.host, cmd], **kwargs)
        except subprocess.CalledProcessError as error:
            if kwargs.get("capture_output"):
                print(error.stderr)
            raise

    def rsync(self, src, dst, opts=(), reverse=False):
        # Prepare: opts
        if isinstance(opts, str):
            opts = opts.split()

        # Prepare: src, dst
        src = str(src)
        dst = str(dst)
        dst = self.host + ":" + dst
        if reverse:
            src, dst = dst, src

        # Get rsync version
        v = (
            subprocess.run(["rsync", "--version"], check=True, text=True, capture_output=True)
            .stdout.splitlines()[0]
            .split()
        )
        i = v.index("version")
        v = v[i + 1]  # => '3.2.3'
        v = [int(w) for w in v.split(".")]
        has_prog2 = (v[0] >= 3) and (v[1] >= 1)

        # Show progress
        progbar = ("--info=progress2", "--no-inc-recursive") if self.progbar and has_prog2 else []

        # Use multiplex
        multiplex = ("-e", self.ssh_M) if self.use_M else []

        # Assemble command
        cmd = ["rsync", "-azhL", *progbar, *multiplex, *opts, src, dst]

        if self.dry:
            # Dry run
            return " ".join(cmd)
        else:
            # Sync
            subprocess.run(cmd, check=True)
            return None

    @contextmanager
    def sym_sync(self, target_dir: Path | str, source_dir: Path, *other):
        """Upload `source_dir` and all `other` to `target_dir` on host. Download upon exit/exception."""
        # Sync source -> target
        self.cmd(f"mkdir -p {target_dir}")
        self.rsync(f"{source_dir}/", target_dir)
        # Sync other.name -> target/
        for p in other:
            p = Path(p).expanduser().resolve()
            if p == Path.home():
                raise ValueError("You probably do not want to sync your entire home dir.")
            self.rsync(f"{p}/", Path(target_dir) / p.name)

        # Reverse sync (i.e. download results) when exiting
        try:
            yield
        finally:
            self.rsync(f"{source_dir}", f"{target_dir}/", reverse=True)
