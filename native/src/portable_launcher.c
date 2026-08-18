// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Approaching AI Authors

#define _GNU_SOURCE

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char* message) {
  fprintf(stderr, "aima-engine launcher: %s: %s\n", message, strerror(errno));
  exit(127);
}

static char* join_path(const char* root, const char* suffix) {
  const size_t root_bytes = strlen(root);
  const size_t suffix_bytes = strlen(suffix);
  char* result = (char*)malloc(root_bytes + suffix_bytes + 1);
  if (result == NULL) die("out of memory");
  memcpy(result, root, root_bytes);
  memcpy(result + root_bytes, suffix, suffix_bytes + 1);
  return result;
}

static void usage(void) {
  fputs(
      "AIMA AMD395 Qwen3.6 35B native engine\n"
      "\n"
      "Usage:\n"
      "  aima-engine --version\n"
      "  aima-engine serve --model-dir PATH [options]\n"
      "  aima-engine resident-session-probe --model-dir PATH [options]\n"
      "  aima-engine tokenizer-probe --model-dir PATH --text TEXT\n"
      "  aima-engine chat-template-probe --model-dir PATH --user TEXT\n"
      "\n"
      "Serve options:\n"
      "  --context-tokens N  Preferred AOT prefill context (default: 8192)\n"
      "  --cache-capacity N   Prompt plus generated-token capacity\n"
      "  --host IPv4          Listen address (default: 127.0.0.1)\n"
      "  --port N             Listen port (default: 8000)\n"
      "  --workers N          Checkpoint reader workers (default: 1)\n"
      "  --chunk-bytes N      Checkpoint read chunk (default: 134217728)\n"
      "  --report PATH        Native weight-load report\n"
      "  --max-requests N     Exit after N successful chat requests\n"
      "\n"
      "Qualified standard contexts:\n"
      "  1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072\n"
      "Qualified window endpoints:\n"
      "  input262143/output1, input261632/output512,\n"
      "  input261120/output1024\n"
      "\n"
      "The model stays resident until SIGINT, SIGTERM, POST /shutdown,\n"
      "or --max-requests. See docs/API.md for the HTTP contract.\n",
      stdout);
}

int main(int argc, char** argv) {
  if (argc == 2 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    usage();
    return 0;
  }
  char executable[PATH_MAX + 1];
  const ssize_t executable_bytes =
      readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (executable_bytes < 0) die("cannot resolve /proc/self/exe");
  if ((size_t)executable_bytes >= sizeof(executable) - 1) {
    errno = ENAMETOOLONG;
    die("launcher path is too long");
  }
  executable[executable_bytes] = '\0';

  char* slash = strrchr(executable, '/');
  if (slash == NULL) {
    errno = EINVAL;
    die("launcher path has no parent directory");
  }
  *slash = '\0';
  slash = strrchr(executable, '/');
  if (slash == NULL) {
    errno = EINVAL;
    die("bundle root cannot be resolved");
  }
  *slash = '\0';

  char* loader = join_path(executable, "/lib/ld-linux-x86-64.so.2");
  char* library_path = join_path(executable, "/lib");
  char* engine = join_path(executable, "/libexec/aima-engine.real");
  if (access(loader, X_OK) != 0) die("bundled ELF loader is missing");
  if (access(engine, X_OK) != 0) die("native engine payload is missing");

  // Run the native payload under the colocated glibc loader. --inhibit-cache
  // makes resolution independent of the host's ld.so.cache, while the static
  // launcher itself has no ELF interpreter or shared-library dependency.
  const size_t fixed_args = 7;
  char** child_argv = (char**)calloc((size_t)argc + fixed_args, sizeof(char*));
  if (child_argv == NULL) die("out of memory");
  child_argv[0] = loader;
  child_argv[1] = "--inhibit-cache";
  child_argv[2] = "--library-path";
  child_argv[3] = library_path;
  child_argv[4] = "--argv0";
  child_argv[5] = argv[0];
  child_argv[6] = engine;
  for (int index = 1; index < argc; ++index) {
    child_argv[index + 6] = argv[index];
  }
  child_argv[argc + 6] = NULL;

  execv(loader, child_argv);
  die("cannot start bundled native engine");
  return 127;
}
