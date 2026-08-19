#!/usr/bin/env python3
"""Run the frozen vLLM API server with official MM timing enabled.

The pinned vLLM exposes ``enable_mm_processor_stats`` in ``EngineArgs`` but
does not register it as an OpenAI-server CLI option.  Its official
``vllm bench mm-processor`` command enables the field with
``parser.set_defaults``; this wrapper applies the same setting and otherwise
delegates parsing, validation, and serving to the original entrypoint.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Approaching AI Authors

from __future__ import annotations

import uvloop

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser


def main() -> None:
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    parser.set_defaults(enable_mm_processor_stats=True)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    if args.enable_mm_processor_stats is not True:
        raise RuntimeError("vLLM multimodal processor timing was not enabled")
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
