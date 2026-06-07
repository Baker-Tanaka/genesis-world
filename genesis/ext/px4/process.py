"""Spawning and teardown of PX4 SITL subprocesses (one per parallel env).

This is only used when ``PX4Options.auto_spawn`` is True. The exact PX4 invocation varies
slightly between PX4 releases; the defaults here follow PX4's own ``sitl_multiple_run.sh``
(separate per-instance working directory, ROMFS at ``<build>/etc``, posix ``rcS`` startup
script). Anything unusual can be injected via ``PX4Options.extra_px4_env``.

PX4's ``simulator_mavlink`` connects out to the simulator's TCP server on port
``4560 + instance``, so with auto-spawn keep ``base_tcp_port = 4560``.
"""

import os
import subprocess

import genesis as gs

from .options import PX4Options


class PX4ProcessManager:
    """Launches and reaps one PX4 process per environment index."""

    def __init__(self, options: PX4Options):
        self._options = options
        self._procs: list[subprocess.Popen] = []
        self._log_files = []

    def spawn(self, instance_id: int) -> subprocess.Popen:
        opt = self._options
        binary = opt.resolved_binary()
        rootfs = opt.rootfs_dir()
        if not os.path.isfile(binary):
            gs.raise_exception(f"PX4 binary not found: {binary}. Build PX4 or set PX4Options.px4_binary.")

        etc_dir = os.path.join(rootfs, "etc")
        rcs = os.path.join("etc", "init.d-posix", "rcS")
        test_data = os.path.join(rootfs, "test_data")

        workdir = os.path.join(rootfs, f"genesis_instance_{instance_id}")
        os.makedirs(workdir, exist_ok=True)

        env = os.environ.copy()
        env["PX4_SIM_MODEL"] = opt.airframe
        env["PX4_SIM_HOSTNAME"] = opt.host
        if opt.sys_autostart is not None:
            env["PX4_SYS_AUTOSTART"] = str(opt.sys_autostart)
        if opt.headless:
            env["HEADLESS"] = "1"
        env.update({k: str(v) for k, v in opt.extra_px4_env.items()})

        cmd = [binary, "-i", str(instance_id), "-d", etc_dir, "-s", rcs]
        if os.path.isdir(test_data):
            cmd += ["-t", test_data]

        log_path = os.path.join(workdir, "px4.log")
        log_file = open(log_path, "w")
        self._log_files.append(log_file)

        gs.logger.info(f"Spawning PX4 instance {instance_id}: {' '.join(cmd)} (log: {log_path})")
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        self._procs.append(proc)
        return proc

    def terminate_all(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self._procs:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._procs.clear()
        for f in self._log_files:
            try:
                f.close()
            except OSError:
                pass
        self._log_files.clear()
