PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m zeroshot.convert \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --method care \
  --rank 256 \
  --dtype fp16 \
  --cal-dataset alpaca \
  --cal-nsamples 256 \
  --cal-max-seqlen 32 \
  --ppl-dataset wikitext2 \
  --output-dir outputs/zero-shot/verify \
  --benchmarks piqa

# arc_easy arc_challenge hellaswag piqa MMLU openbookqa race winogrande 