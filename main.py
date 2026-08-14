"""FindDogBreed CLI entrypoint.

실행 제어만 담당한다. 기능 구현과 argparse 정의는 다른 모듈에 둔다.
"""

from __future__ import annotations

import sys

from cli import build_parser


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_console_encoding()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
