#!/usr/bin/env bash

care_setup_runtime() {
  local project_dir="$1"

  export PYTHONPATH="${project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
  fi

  if [[ -n "${CONDA_ENV:-}" ]]; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "${CONDA_ENV}"
  fi
}

normalize_dataset_name() {
  case "$1" in
    wiki|wikitext)
      printf '%s\n' "wikitext2"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}
