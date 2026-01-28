from pathlib import Path

def define_env(env):
    # Base path is always the docs folder where this macro lives
    base_path = Path(__file__).parent.resolve()

    @env.macro
    def read_file(rel_path: str) -> str:
        """
        Read the content of a file relative to the docs folder.
        Works locally and in CI/CD builds, regardless of working directory.
        """
        # Make sure we resolve the full path
        file_path = (base_path / rel_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return file_path.read_text(encoding="utf-8")
