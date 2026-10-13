# Local Ollama Model Benchmark

Date: 2026-10-13

## Environment

- CPU: AMD Ryzen 7 7735HS, 8 cores / 16 logical processors
- RAM: 16 GB installed, 13.3 GB visible to Windows; 0.8 GB free during measurement
- GPU: integrated AMD Radeon 680M
- Ollama models: `llama3.2:3b` (2.02 GB), `qwen3:0.6b` (523 MB), `qwen3:1.7b` (1.36 GB)

## Structured Output Timing

Each model received the same short AgentReport prompt and JSON schema. Cold time is one request after unloading the model; warm time is the next request with the model retained. These are single observations, not statistical averages.

| Model | Cold total | Warm total | JSON schema parsed | Cold load time |
|---|---:|---:|---|---:|
| `llama3.2:3b` | 27.92 s | 5.98 s | Yes | 22.01 s |
| `qwen3:0.6b` | 10.84 s | 5.14 s | Yes | 7.12 s |
| `qwen3:1.7b` | 39.81 s | 6.25 s | Yes | 33.95 s |

## Validator Tool Check

For the standalone Ollama tool-call test, each model received the same `validate_portfolio` schema and a valid ten-symbol target portfolio. The emitted arguments were checked and, where valid, passed to the project validator with `dry_run=true`; this used its local demo account and made no Alpaca request.

| Model | Native tool call | Arguments usable | Dry validator result |
|---|---|---|---|
| `llama3.2:3b` | Yes | Boolean values were strings; CrewAI tool schema coerced them successfully | Not invoked in this standalone comparison |
| `qwen3:0.6b` | Yes | Yes | Passed all six checks |
| `qwen3:1.7b` | Yes | Yes | Passed all six checks |

## Combined/Actual CrewAI Check

The same request was then sent with both the AgentReport JSON format and the validator tool schema. All three models returned valid JSON but emitted no tool call. A minimal CrewAI RiskOfficer task also emitted no tool call for any model, even with three agent iterations. Therefore none currently passes the combined structured-output plus tool-use requirement in this project's CrewAI execution path.

## Decision

`qwen3:0.6b` is fastest in this single-run sample: 10.84 s cold and 5.14 s warm, versus 27.92 s / 5.98 s for the configured `llama3.2:3b`. It passes standalone Ollama tool calling and the local dry validator. However, no candidate called the tool when structured response formatting was enabled or through the actual CrewAI agent. The model configs were not changed: switching to the faster model would not resolve the integration failure and would falsely suggest RiskOfficer had invoked validation.

After the latency tradeoff was raised, all three student configs were changed to `qwen3:0.6b`, preserving the provider fallback order. The cycle runner continues to gate execution on the deterministic MCP validation result, independent of the LLM's decision. The model still does not demonstrate validator invocation through CrewAI when structured response formatting is enabled; do not treat its narrative report as proof that the tool ran.

No Alpaca credentials were loaded, no market APIs were called, and no orders were submitted during this benchmark.
