from __future__ import annotations

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "git is required to apply the LoRA patch")
class LoraPatchTests(unittest.TestCase):
    def test_new_python_files_apply_completely_and_parse(self) -> None:
        patch = (ROOT / "patches" / "0001-qwen3-tts-lora.patch").read_text(encoding="utf-8")
        # Existing-file hunks require a Qwen3-TTS checkout; new files can be
        # checked independently without model dependencies or GPU hardware.
        sections = [
            "diff --git " + section
            for section in patch.split("diff --git ")[1:]
            if "\n--- /dev/null\n" in section
        ]
        self.assertTrue(sections, "the patch must contain new LoRA Python files")
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True, capture_output=True)
            # Use the same ordinary git apply as scripts/apply_patches.sh.
            # --recount would hide stale hunk counts and truncated file output.
            applied = subprocess.run(
                ["git", "-C", str(checkout), "apply", "-"],
                input="".join(sections),
                text=True,
                encoding="utf-8",
                capture_output=True,
            )
            self.assertEqual(applied.returncode, 0, applied.stderr)
            for section in sections:
                relative = next(line[6:] for line in section.splitlines() if line.startswith("+++ b/"))
                if not relative.endswith(".py"):
                    continue
                with self.subTest(path=relative):
                    source = (checkout / relative).read_text(encoding="utf-8")
                    expected = "\n".join(
                        line[1:]
                        for line in section.splitlines()
                        if line.startswith("+") and not line.startswith("+++ ")
                    ) + "\n"
                    self.assertEqual(source, expected, "git apply must preserve every added line")
                    ast.parse(source, filename=relative)


if __name__ == "__main__":
    unittest.main()
