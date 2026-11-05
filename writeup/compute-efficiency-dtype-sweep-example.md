# Dtype sweep: what quantization actually costs and saves (measured, not assumed)

Everything else in this project's write-ups tests the *offload* lever.
Quantization is the other lever this project has real code for
(`BaselinePolicy`/`OffloadPolicy` loading real `bitsandbytes` int8/int4 —
see ROADMAP.md's "Real quantized loading" entry) but, until this sweep, it
had only been confirmed to *load and generate correctly*, never actually
benchmarked for its speed/memory tradeoff.

```bash
python -m tron_efficiency.cli dtype-sweep --model EleutherAI/pythia-70m \
    --max-new-tokens 16 --dtypes fp32,fp16,int8,int4
```

**Real result** (CPU, this dev machine, `pythia-70m`):

| Dtype | Load (s) | TTFT (s) | E2E latency (s) | Throughput (tok/s) | Peak CPU memory |
|---|---|---|---|---|---|
| fp32 | 16.05 | 0.395 | 1.036 | 15.45 | 804MB |
| fp16 | 4.75 | 0.376 | 0.903 | 17.71 | 783MB |
| int8 | 15.79 | 0.450 | 2.653 | **6.03** | 736MB |
| int4 | 9.88 | 0.166 | 1.440 | 11.11 | **589MB** |

**What this actually shows, and why it matters more than a clean story
would:**

- **Memory drops monotonically and genuinely** as precision drops — 804MB
  (fp32) → 589MB (int4), a real **27% reduction**. This is the win
  quantization is supposed to deliver, and it's real here.
- **Throughput does NOT drop monotonically** — and this is the finding
  worth taking seriously: **int8 is 2.6x *slower* than fp32** (6.03 vs.
  15.45 tok/s), not faster, despite using less memory. int4 does better
  than int8 but still doesn't beat fp16. fp16 is the actual speed winner
  here, not either quantized option.
- **The likely cause, stated as a hypothesis, not confirmed:** bitsandbytes'
  int8/int4 CPU kernels are not necessarily optimized the way native
  PyTorch fp32/fp16 operations are — on real GPU hardware with tensor
  cores, int8 quantization typically *does* deliver a speed win alongside
  the memory win; on CPU, that hardware acceleration doesn't exist, so
  quantization overhead (packing/unpacking, dequantizing for compute) can
  dominate. This project has not yet benchmarked dtype sweep on real GPU
  hardware to confirm this — it's a reasonable explanation for what was
  measured, not an independently verified cause.
- **Output quality is visibly affected, especially at int4**: fp16 and
  int8 happened to produce byte-identical text here; int4's output
  diverges more noticeably ("the dog's name is 'Bare'..." vs. the
  others' more coherent continuations) — consistent with int4 being the
  most aggressive, lossiest option. This project does not have an
  automated quality metric (perplexity, task accuracy) to quantify this
  beyond eyeballing the text — a real limitation, not glossed over.

**Why this matters for the "quantization is a lever" pitch:** it's a real
lever, but *not a free one on every hardware target*, and this is exactly
the kind of claim that would have been wrong to make without measuring it.
"Quantize to save memory and get a speed bonus" is the GPU story; on CPU,
the honest story measured here is "quantize to save memory, possibly at a
real speed cost, and verify before promising both." A product recommending
quantization should check which hardware it's targeting before claiming
both wins — this is now a real, measured data point informing that, not
an assumption.

**Honest scope:** one small model (`pythia-70m`), one CPU machine, one run
per dtype (no repeated trials). The GPU-side version of this experiment
(does int8 actually deliver the expected speed win there) is a natural
next step, not yet run.
