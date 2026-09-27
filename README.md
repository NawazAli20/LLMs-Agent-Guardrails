![LLM Guardrails & Security — Build Safer AI Agents](output/middleware-slides/llm-guardrails-youtube-banner.png)

# LLM Guardrails & Security

### Build safer, more reliable AI agents with LangChain

A hands-on lecture on adding practical guardrails to an LLM agent: recover from model failures, bound retries, and control model and tool usage.

The examples use an agent with a weather tool and Tavily search, with middleware to keep its execution predictable.

## What you will learn

| Middleware | What it does |
| --- | --- |
| **ModelFallbackMiddleware** | Switches to backup models when a model call fails. |
| **ModelCallLimitMiddleware** | Limits model calls per run or conversation to control usage and prevent runaway loops. |
| **ModelRetryMiddleware** | Retries failed model requests with configurable delays and a maximum retry limit. |
| **ToolCallLimitMiddleware** | Limits tool calls, such as weather lookups and Tavily searches, to manage API usage. |

These controls address reliability and resource usage. They are one part of an LLM security strategy; they do not by themselves prevent prompt injection or validate generated answers.

## Add the guardrails

The following configuration builds on the lecture code. It assumes that `model`, `model_fallback1`, `model_fallback2`, `get_weather`, and `primary_internet_search` have already been initialized.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[get_weather, primary_internet_search],
    checkpointer=InMemorySaver(),
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=3,
            exit_behavior="end",
        ),
        ModelRetryMiddleware(
            max_retries=5,
            backoff_factor=1.5,
            initial_delay=1,
            max_delay=60,
        ),
        ModelFallbackMiddleware(
            model_fallback1,
            model_fallback2,
        ),
        ToolCallLimitMiddleware(
            tool_name=get_weather.name,
            run_limit=2,
            exit_behavior="continue",
        ),
        ToolCallLimitMiddleware(
            tool_name=primary_internet_search.name,
            run_limit=3,
            exit_behavior="continue",
        ),
    ],
)

# Reuse this thread ID to retain conversation state and thread counts.
config = {"configurable": {"thread_id": "guardrails-demo"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in Peoria?"}]},
    config,
)
print(result["messages"][-1].content)
```

The weather and search budgets above are illustrative additions to the lecture's original configuration. Use each tool's `.name` property to target its registered name.

## Understand the limits

- **Per run:** `run_limit` applies to one agent invocation and resets on the next invocation.
- **Per conversation:** `thread_limit` accumulates across invocations with the same thread ID and requires a checkpointer. `InMemorySaver` retains this state within the running process.
- **Retries:** `max_retries=5` allows the initial attempt plus up to five retries. Actual waits vary when jitter is enabled.
- **Middleware order:** In this example, retry wraps fallback. If the complete fallback chain raises an error eligible for retry, the retry middleware can rerun the chain.
- **Model calls versus API attempts:** The model-call budget counts model steps in the agent loop; retries and fallback can add underlying API attempts.
- **Stopping behavior:** The model limiter uses `"end"` to stop the run gracefully. The tool limiters use `"continue"` to block excess tool calls while letting the agent continue.

## Try it in the lecture

| Demo | What to observe |
| --- | --- |
| Simulate a primary-model failure | The fallback models are tried in their configured order. |
| Simulate two retryable timeouts | The retry middleware waits between attempts, then allows a successful call through. |
| Request a fourth model step in one run | The per-run model budget prevents the next model call. |
| Attempt a third weather call or fourth Tavily search in one run | The corresponding tool limiter blocks the excess call. |

For a repeatable retry demo, use a temporary `wrap_model_call` hook that raises `TimeoutError` on the first two attempts. Put it after `ModelRetryMiddleware`, set `retry_on=(TimeoutError,)` and `jitter=False`, and test it in a separate agent without fallback. To demonstrate exhaustion, keep raising the error and set `on_failure="error"`.

## Lecture slides

| Topic | Slide image |
| --- | --- |
| Model fallback | [View slide](output/middleware-slides/model-fallback.png) |
| Model call limits | [View slide](output/middleware-slides/model-call-limit.png) |
| Model retry | [View slide](output/middleware-slides/model-retry.png) |
| Tool call limits | [View slide](output/middleware-slides/tool-call-limit.png) |

## References

- [LangChain prebuilt middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- [Custom middleware and execution order](https://docs.langchain.com/oss/python/langchain/middleware/custom)

---

**Build agents that recover from failures and operate within clear usage limits.**
