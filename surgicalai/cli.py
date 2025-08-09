import argparse

from .logging_utils import get_logger

logger = get_logger(__name__)


def app():
    parser = argparse.ArgumentParser(prog="surgicalai")
    sub = parser.add_subparsers(dest="command")

    for cmd in ["demo", "ingest", "analyze", "plan", "visualize"]:
        sub.add_parser(cmd)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    logger.info("run", command=args.command)
    print(f"Executed {args.command} (stub)")


if __name__ == "__main__":
    app()
