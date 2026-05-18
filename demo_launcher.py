from __future__ import annotations

import argparse
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from vaultwares_studio.pipeline import DEFAULT_SOURCE_VIDEO, DigitalTwinStudioRunner, create_job_manifest

APP_ROOT = Path(__file__).resolve().parent


class DemoLauncherUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Digital Twin Studio Launcher")
        self.root.geometry("980x640")

        self.current_manifest = create_job_manifest(DEFAULT_SOURCE_VIDEO)
        self.run_thread: threading.Thread | None = None
        self.status_text = tk.StringVar(value="Ready")

        self._build()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(frame)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Run Full Demo", command=self.start_demo).pack(side=tk.LEFT)
        ttk.Button(controls, text="Choose Video", command=self.choose_video).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Open Job Folder", command=self.open_job_folder).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="Open Walkthrough", command=self.open_walkthrough).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(frame, textvariable=self.status_text).pack(anchor=tk.W, pady=(10, 6))
        self.log_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=30)
        self.log_widget.pack(fill=tk.BOTH, expand=True)

        self.log(f"Source video: {self.current_manifest.source_video}")
        self.log(f"Output folder: {self.current_manifest.output_dir}")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_widget.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_widget.see(tk.END)

    def choose_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choose input video",
            initialdir=str(APP_ROOT),
            filetypes=[("Video Files", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All Files", "*.*")],
        )
        if file_path:
            self.current_manifest = create_job_manifest(Path(file_path))
            self.log(f"Created new job {self.current_manifest.job_id} for {file_path}")

    def start_demo(self) -> None:
        if self.run_thread and self.run_thread.is_alive():
            return

        def run() -> None:
            self.status_text.set("Running guided pipeline...")
            try:
                runner = DigitalTwinStudioRunner(self.current_manifest, self.log)
                self.current_manifest = runner.run_remaining()
                self.status_text.set("Demo finished")
                self.log(f"Walkthrough video: {self.current_manifest.walkthrough_video}")
            except Exception as exc:  # noqa: BLE001
                self.status_text.set("Demo failed")
                self.log(f"[ERROR] {exc}")
                self.root.after(0, lambda: messagebox.showerror("Demo failed", str(exc)))

        self.run_thread = threading.Thread(target=run, daemon=True)
        self.run_thread.start()

    def open_job_folder(self) -> None:
        subprocess.Popen(["explorer", self.current_manifest.output_dir])

    def open_walkthrough(self) -> None:
        if not self.current_manifest.walkthrough_video:
            self.log("No walkthrough video generated yet.")
            return
        subprocess.Popen(["explorer", self.current_manifest.walkthrough_video])

    def run(self) -> None:
        self.root.mainloop()


def run_headless(source_video: Path) -> int:
    manifest = create_job_manifest(source_video)
    runner = DigitalTwinStudioRunner(manifest, print)
    result = runner.run_remaining()
    print(f"Job manifest: {Path(result.output_dir) / 'manifest.json'}", flush=True)
    print(f"Walkthrough video: {result.walkthrough_video}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Digital Twin Studio pipeline locally.")
    parser.add_argument("--headless", action="store_true", help="Run the full pipeline without Tk.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_VIDEO), help="Input video path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    if args.headless:
        return run_headless(source)
    app = DemoLauncherUI()
    if source != Path(DEFAULT_SOURCE_VIDEO).resolve():
        app.current_manifest = create_job_manifest(source)
        app.log(f"Created new job {app.current_manifest.job_id} for {source}")
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
