"""Gahan AI Validation Viewer.

  python app.py                 desktop app
  python app.py --cli --help    headless run (save annotated video, exports, summary)
"""
import sys


def main():
    if "--cli" in sys.argv:
        argv = [a for a in sys.argv[1:] if a != "--cli"]
        from core.cli import main as cli_main
        sys.exit(cli_main(argv))
    from ui.main_window import run_app
    sys.exit(run_app(sys.argv))


if __name__ == "__main__":
    main()
