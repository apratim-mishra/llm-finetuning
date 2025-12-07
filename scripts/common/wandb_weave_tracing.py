#!/usr/bin/env python3
"""
Wandb Weave Tracing Integration for LLM Fine-tuning Pipeline

Wandb Weave provides:
- Automatic tracing of LLM calls
- Token usage and cost tracking
- Latency monitoring
- Dataset management and evaluation
- Model versioning and lineage

Usage:
    # Initialize tracing in your script
    from scripts.common.wandb_weave_tracing import init_weave, trace_llm_call

    init_weave("my-project")

    # Trace inference calls
    @trace_llm_call
    def generate_response(model, prompt):
        return model.generate(prompt)

    # Or use the LLMTracer class
    tracer = LLMTracer("my-project")
    with tracer.trace("inference"):
        result = model.generate(prompt)
"""

import functools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Check for weave availability
WEAVE_AVAILABLE = False
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    logger.warning("Wandb Weave not installed. Install with: pip install weave")


@dataclass
class TraceMetrics:
    """Metrics collected during a traced operation."""
    operation: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Calculate final metrics."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.total_tokens = self.input_tokens + self.output_tokens


def init_weave(project_name: str = "llm-finetuning") -> bool:
    """
    Initialize Wandb Weave tracing.

    Args:
        project_name: Name of the Weave project

    Returns:
        True if initialization successful, False otherwise
    """
    if not WEAVE_AVAILABLE:
        logger.warning("Weave not available. Tracing disabled.")
        return False

    try:
        weave.init(project_name)
        logger.info(f"Wandb Weave initialized for project: {project_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Weave: {e}")
        return False


def trace_llm_call(func: Optional[Callable] = None, *, name: Optional[str] = None):
    """
    Decorator to trace LLM calls with Wandb Weave.

    Usage:
        @trace_llm_call
        def generate(prompt):
            ...

        @trace_llm_call(name="custom_name")
        def my_function():
            ...
    """
    def decorator(fn: Callable) -> Callable:
        op_name = name or fn.__name__

        if WEAVE_AVAILABLE:
            # Use weave.op() decorator
            traced_fn = weave.op()(fn)
            return traced_fn
        else:
            # Fallback: basic timing without Weave
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = fn(*args, **kwargs)
                    duration = (time.time() - start) * 1000
                    logger.debug(f"{op_name} completed in {duration:.2f}ms")
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    logger.error(f"{op_name} failed after {duration:.2f}ms: {e}")
                    raise
            return wrapper

    if func is not None:
        return decorator(func)
    return decorator


class LLMTracer:
    """
    Context manager and utility class for tracing LLM operations.

    Usage:
        tracer = LLMTracer("my-project")

        # As context manager
        with tracer.trace("inference") as t:
            result = model.generate(prompt)
            t.set_tokens(input=100, output=50)

        # Log custom metrics
        tracer.log_metrics({"accuracy": 0.95, "loss": 0.05})

        # Log evaluation results
        tracer.log_evaluation(predictions, references, metrics)
    """

    def __init__(self, project_name: str = "llm-finetuning", enabled: bool = True):
        self.project_name = project_name
        self.enabled = enabled and WEAVE_AVAILABLE
        self.traces: List[TraceMetrics] = []
        self._current_trace: Optional[TraceMetrics] = None

        if self.enabled:
            init_weave(project_name)

    def trace(self, operation: str, model_name: str = ""):
        """Create a trace context for an operation."""
        return _TraceContext(self, operation, model_name)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to Weave."""
        if not self.enabled:
            logger.info(f"Metrics (not logged): {metrics}")
            return

        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Dict[str, float],
        dataset_name: str = "evaluation",
    ):
        """
        Log evaluation results to Weave.

        Creates a Weave dataset and logs predictions with metrics.
        """
        if not self.enabled:
            logger.info(f"Evaluation metrics: {metrics}")
            return

        try:
            # Create evaluation data
            eval_data = []
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                eval_data.append({
                    "id": i,
                    "prediction": pred,
                    "reference": ref,
                })

            # Log as Weave dataset if available
            if WEAVE_AVAILABLE:
                dataset = weave.Dataset(name=dataset_name, rows=eval_data)
                weave.publish(dataset)

            # Also log summary metrics
            self.log_metrics({f"eval/{k}": v for k, v in metrics.items()})

        except Exception as e:
            logger.warning(f"Failed to log evaluation: {e}")

    def log_model(
        self,
        model_path: str,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log model artifact to Weave."""
        if not self.enabled:
            logger.info(f"Model logged (locally): {model_name} at {model_path}")
            return

        try:
            import wandb
            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name=model_name,
                    type="model",
                    metadata=metadata or {},
                )
                artifact.add_dir(model_path)
                wandb.log_artifact(artifact)
                logger.info(f"Model artifact logged: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to log model: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all traces."""
        if not self.traces:
            return {}

        total_duration = sum(t.duration_ms for t in self.traces)
        total_tokens = sum(t.total_tokens for t in self.traces)
        success_rate = sum(1 for t in self.traces if t.success) / len(self.traces)

        return {
            "total_traces": len(self.traces),
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / len(self.traces),
            "total_tokens": total_tokens,
            "success_rate": success_rate,
        }

    def save_traces(self, output_path: str):
        """Save traces to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        traces_data = [
            {
                "operation": t.operation,
                "duration_ms": t.duration_ms,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "model_name": t.model_name,
                "success": t.success,
                "error": t.error,
                "metadata": t.metadata,
            }
            for t in self.traces
        ]

        with open(output_path, 'w') as f:
            json.dump({
                "traces": traces_data,
                "summary": self.get_summary(),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        logger.info(f"Traces saved to: {output_path}")


class _TraceContext:
    """Context manager for individual traces."""

    def __init__(self, tracer: LLMTracer, operation: str, model_name: str = ""):
        self.tracer = tracer
        self.metrics = TraceMetrics(
            operation=operation,
            start_time=time.time(),
            model_name=model_name,
        )

    def __enter__(self):
        self.tracer._current_trace = self.metrics
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.finalize()

        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error = str(exc_val)

        self.tracer.traces.append(self.metrics)
        self.tracer._current_trace = None

        # Log to Weave if enabled
        if self.tracer.enabled:
            self.tracer.log_metrics({
                f"trace/{self.metrics.operation}/duration_ms": self.metrics.duration_ms,
                f"trace/{self.metrics.operation}/tokens": self.metrics.total_tokens,
            })

        return False  # Don't suppress exceptions

    def set_tokens(self, input: int = 0, output: int = 0):
        """Set token counts for this trace."""
        self.metrics.input_tokens = input
        self.metrics.output_tokens = output

    def set_metadata(self, **kwargs):
        """Set additional metadata."""
        self.metrics.metadata.update(kwargs)


# Weave-enabled model wrapper
class TracedModel:
    """
    Wrapper to add Weave tracing to any model.

    Usage:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("...")
        traced_model = TracedModel(model, "my-model")
        output = traced_model.generate(input_ids)
    """

    def __init__(self, model: Any, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.tracer = LLMTracer()

    def __getattr__(self, name: str):
        """Proxy attribute access to underlying model."""
        attr = getattr(self.model, name)

        if callable(attr) and name in ["generate", "forward", "__call__"]:
            @functools.wraps(attr)
            def traced_method(*args, **kwargs):
                with self.tracer.trace(f"{self.model_name}.{name}", self.model_name):
                    return attr(*args, **kwargs)
            return traced_method

        return attr


# Convenience functions for common operations
@trace_llm_call(name="vllm_inference")
def trace_vllm_generate(llm, prompts: List[str], sampling_params) -> List:
    """Traced vLLM generation."""
    return llm.generate(prompts, sampling_params)


@trace_llm_call(name="sglang_inference")
def trace_sglang_generate(runtime, prompts: Union[str, List[str]], sampling_params: Dict) -> Any:
    """Traced SGLang generation."""
    return runtime.generate(prompts, sampling_params=sampling_params)


@trace_llm_call(name="hf_inference")
def trace_hf_generate(model, tokenizer, prompts: List[str], **generate_kwargs) -> List[str]:
    """Traced HuggingFace generation."""
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, **generate_kwargs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def create_evaluation_callback(tracer: LLMTracer):
    """
    Create a callback for training that logs to Weave.

    Usage:
        tracer = LLMTracer("my-project")
        trainer = SFTTrainer(
            ...,
            callbacks=[create_evaluation_callback(tracer)]
        )
    """
    try:
        from transformers import TrainerCallback

        class WeaveCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    tracer.log_metrics(logs, step=state.global_step)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics:
                    tracer.log_metrics(
                        {f"eval/{k}": v for k, v in metrics.items()},
                        step=state.global_step
                    )

            def on_train_end(self, args, state, control, **kwargs):
                tracer.save_traces(f"{args.output_dir}/weave_traces.json")

        return WeaveCallback()

    except ImportError:
        logger.warning("transformers not installed, callback not available")
        return None


if __name__ == "__main__":
    # Example usage
    print("Wandb Weave Tracing Integration")
    print("================================")

    # Initialize
    tracer = LLMTracer("llm-finetuning-demo")

    # Simulate some traces
    with tracer.trace("test_operation", "test-model") as t:
        time.sleep(0.1)  # Simulate work
        t.set_tokens(input=100, output=50)
        t.set_metadata(prompt_type="instruction")

    with tracer.trace("another_operation") as t:
        time.sleep(0.05)
        t.set_tokens(input=200, output=100)

    # Get summary
    summary = tracer.get_summary()
    print(f"\nTrace Summary:")
    print(f"  Total traces: {summary['total_traces']}")
    print(f"  Total duration: {summary['total_duration_ms']:.2f}ms")
    print(f"  Total tokens: {summary['total_tokens']}")
    print(f"  Success rate: {summary['success_rate']:.1%}")

    # Save traces
    tracer.save_traces("outputs/traces/demo_traces.json")
