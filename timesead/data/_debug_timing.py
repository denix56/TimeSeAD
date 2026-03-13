import collections
import logging
import os
import time
from typing import Any, Callable

import torch

_DEBUG_LOG_FIRST_N = 3
_DEBUG_LOG_EVERY_N = 512
_DEBUG_LOG_SLOW_SECS = 0.05
_stage_call_counts = collections.Counter()
_configured_logging_pids = set()


def _next_stage_call_idx(stage: str) -> int:
    _stage_call_counts[stage] += 1
    return _stage_call_counts[stage]


def _should_log(call_idx: int, elapsed_seconds: float) -> bool:
    return (
        call_idx <= _DEBUG_LOG_FIRST_N
        or call_idx % _DEBUG_LOG_EVERY_N == 0
        or elapsed_seconds > _DEBUG_LOG_SLOW_SECS
    )


def _resolve_debug_value(value: Any, result: Any) -> Any:
    if callable(value):
        return value(result)

    return value


def _summarize_shapes(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        return tuple(value.shape)

    if isinstance(value, torch.Size):
        return tuple(value)

    if isinstance(value, tuple):
        if all(isinstance(dim, int) for dim in value):
            return tuple(value)

        return tuple(_summarize_shapes(item) for item in value)

    if isinstance(value, list):
        if not value:
            return []

        return {'len': len(value), 'first': _summarize_shapes(value[0])}

    if isinstance(value, dict):
        return {key: _summarize_shapes(item) for key, item in value.items()}

    return type(value).__name__


def _format_extra(extra: Any) -> str:
    if not extra:
        return ''

    if not isinstance(extra, dict):
        return f' extra={extra}'

    return ' ' + ' '.join(f'{key}={value}' for key, value in extra.items())


def _ensure_process_logging(log_level: int) -> None:
    pid = os.getpid()
    if pid in _configured_logging_pids:
        return

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s %(levelname)s %(name)s %(message)s',
        )

    _configured_logging_pids.add(pid)


def run_with_debug_timing(
    logger: logging.Logger,
    stage: str,
    fn: Callable[[], Any],
    *,
    index_label: str,
    index_value: Any,
    input_value: Any = None,
    output_value: Any = None,
    extra: Any = None,
    log_level: int = logging.DEBUG,
    initialize_logging: bool = False,
) -> Any:
    if initialize_logging:
        _ensure_process_logging(log_level)

    if not logger.isEnabledFor(log_level):
        return fn()

    call_idx = _next_stage_call_idx(stage)
    start = time.perf_counter()
    result = fn()
    elapsed_seconds = time.perf_counter() - start

    if _should_log(call_idx, elapsed_seconds):
        worker_info = torch.utils.data.get_worker_info()
        debug_input = _resolve_debug_value(input_value, result)
        debug_output = result if output_value is None else _resolve_debug_value(output_value, result)
        debug_extra = _resolve_debug_value(extra, result)

        logger.log(
            log_level,
            '%s %s=%s elapsed=%.6fs pid=%s worker_id=%s call_idx=%s input_shapes=%s output_shapes=%s%s',
            stage,
            index_label,
            index_value,
            elapsed_seconds,
            os.getpid(),
            worker_info.id if worker_info is not None else None,
            call_idx,
            _summarize_shapes(debug_input),
            _summarize_shapes(debug_output),
            _format_extra(debug_extra),
        )

    return result
