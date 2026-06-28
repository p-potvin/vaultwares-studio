from pathlib import Path
from vaultwares_studio.pipeline import create_job_manifest, DEFAULT_CAMERA_PROMPT, DigitalTwinStudioRunner
from vaultwares_studio.runners.hf_jobs import HfJobsStageRunner, HfJobsConfig

source_video = Path("inputs/cloudyday1_june14_194sec.MOV").resolve()
manifest = create_job_manifest(source_video, DEFAULT_CAMERA_PROMPT)
config = HfJobsConfig.load()
remote_runner = HfJobsStageRunner(config=config, confirm_cost=lambda _e: True) if config.enabled else None

def log(msg):
    print(msg, flush=True)

runner = DigitalTwinStudioRunner(manifest, log, strict_mode=False, remote_runner=remote_runner)
runner.run_remaining()
