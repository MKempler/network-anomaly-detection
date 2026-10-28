"""Read-only investigation tools over one case's observable telemetry.

An `Investigation` is one flagged event: a case plus the time of the flag. The
tools compare a flagged window (just after the flag) with a baseline (just
before it) for metrics, logs and traces. Rankers and the agent see nothing
else.

Leakage rules this module enforces:
- it never receives labels; the anchor is just a time, and the caller decides
  whether that time came from a detector or from the benchmark's inject time
- every time it reports is relative to the flag ("+45 s"), never absolute
- log messages are reduced to templates (numbers, ids and addresses masked),
  which also strips the absolute timestamps some systems print inside messages

Every tool result carries `evidence`: uniform (service, signal, kind, score)
items. `verify.py` checks a ranked cause against the evidence of the calls it
cites, so every kind of signal is checked the same way.
"""

import inspect
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Investigation as InvestigationConfig
from src.load_rcaeval import Case, load_logs, load_traces

MASKS = [
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<id>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<ip>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<id>"),
    (re.compile(r"\d+(?:[.:/-]\d+)*"), "<n>"),
]
LEVEL = re.compile(r"\b(?:ERROR|FATAL|SEVERE|WARN(?:ING)?)\b", re.I)


def template(message: str, limit: int = 160) -> str:
    """Mask numbers, ids and addresses so repeated messages collapse into one pattern."""
    for pattern, token in MASKS:
        message = pattern.sub(token, message)
    return re.sub(r"\s+", " ", message).strip()[:limit]


def templates(messages: pd.Series, limit: int = 160) -> pd.Series:
    """`template` for a whole column at once (vectorised; Train Ticket has ~280k lines per case)."""
    out = messages.astype("str")
    for pattern, token in MASKS:
        out = out.str.replace(pattern, token, regex=True)
    return out.str.replace(r"\s+", " ", regex=True).str.strip().str.slice(0, limit)


def sig(x: float, digits: int = 4) -> float:
    """Round to significant digits; keeps tool output short."""
    if x == 0 or not math.isfinite(x):
        return 0.0 if not math.isfinite(x) else x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict
    result: dict
    evidence: list[dict] = field(default_factory=list)


class Investigation:
    def __init__(self, case: Case, anchor: float, cfg: InvestigationConfig, logs=None, traces=None):
        self.case, self.anchor, self.cfg = case, float(anchor), cfg
        self._logs_source = logs if logs is not None else (lambda: load_logs(case.case_id))
        self._traces_source = traces if traces is not None else (lambda: load_traces(case.case_id))
        self._deviations = self._logs = self._traces = None
        self.calls: list[ToolCall] = []

    # -- shared helpers -------------------------------------------------------

    def _phase(self, times: pd.Series | pd.Index) -> np.ndarray:
        """'baseline', 'window' or '' for each timestamp."""
        t = np.asarray(times, dtype=float)
        out = np.full(len(t), "", dtype=object)
        out[(t >= self.anchor - self.cfg.baseline_s) & (t < self.anchor)] = "baseline"
        out[(t >= self.anchor) & (t < self.anchor + self.cfg.window_s)] = "window"
        return out

    def _floor(self, std: np.ndarray, *levels: np.ndarray) -> np.ndarray:
        """Baseline std, floored relative to the metric's own size.

        A metric that is flat through the baseline has std ~0, and dividing by
        that turns a tiny move into a score of 10^5, burying real signals. The
        floor is a fraction of the larger of the baseline and window levels, so a
        flat metric that moves scores at most about 2 / rel_std_floor.
        """
        scale = np.max(np.abs(np.vstack(levels)), axis=0)
        return np.maximum.reduce([std, self.cfg.rel_std_floor * scale, np.full_like(std, self.cfg.abs_std_floor)])

    def trace_service(self, name: str) -> str:
        """Map a trace's service name onto the metric service names (frontendservice -> frontend)."""
        candidates = set(self.case.candidate_services)
        for option in (name, name.removesuffix("service"), name.removesuffix("-service")):
            if option in candidates:
                return option
        return name

    # -- metrics ---------------------------------------------------------------

    def metric_deviations(self) -> pd.DataFrame:
        """Per metric: window vs baseline, with a z-like score. Sorted, largest first."""
        if self._deviations is None:
            m = self.case.metrics
            phase = self._phase(m.index)
            base, win = m[phase == "baseline"], m[phase == "window"]
            b_mean, b_std = base.mean().to_numpy(), base.std().fillna(0).to_numpy()
            w_mean = win.mean().to_numpy()
            score = np.abs(w_mean - b_mean) / self._floor(b_std, b_mean, w_mean)
            constant = (base.nunique().to_numpy() <= 1) & (win.nunique().to_numpy() <= 1) & (w_mean == b_mean)
            score[constant | np.isnan(score)] = 0.0
            # Flat metrics that move all reach the floor's ceiling; break those ties by
            # relative change so the order isn't just column order.
            relative = np.abs(w_mean - b_mean) / np.maximum(np.maximum(np.abs(b_mean), np.abs(w_mean)),
                                                            self.cfg.abs_std_floor)
            self._deviations = pd.DataFrame({
                "metric": m.columns, "service": [self.case.services[c] for c in m.columns],
                "score": score, "relative_change": relative, "baseline_mean": b_mean, "baseline_std": b_std,
                "window_mean": w_mean, "window_max": win.max().to_numpy(), "window_min": win.min().to_numpy(),
            }).sort_values(["score", "relative_change"], ascending=False, kind="stable").reset_index(drop=True)
        return self._deviations

    def top_deviations(self, k: int | None = None, service: str | None = None) -> dict:
        d = self.metric_deviations()
        if service is not None:
            d = d[d["service"] == service]
        rows = d.head(k or self.cfg.top_k)
        items = [{"metric": r.metric, "service": r.service, "score": sig(r.score),
                  "direction": "up" if r.window_mean >= r.baseline_mean else "down",
                  "baseline_mean": sig(r.baseline_mean), "window_mean": sig(r.window_mean)} for r in rows.itertuples()]
        evidence = [{"service": i["service"], "signal": i["metric"], "kind": "metric", "score": i["score"]}
                    for i in items]
        return {"metrics": items, "evidence": evidence}

    def compare_to_baseline(self, metric: str) -> dict:
        d = self.metric_deviations()
        if metric not in self.case.services:
            return {"error": f"unknown metric {metric!r}; list names with top_deviations"}
        r = d[d["metric"] == metric].iloc[0]
        series = self.case.metrics[metric]
        win = series[self._phase(series.index) == "window"]
        std = float(self._floor(np.array([r.baseline_std]), np.array([r.baseline_mean]), np.array([r.window_mean]))[0])
        over = win[np.abs(win - r.baseline_mean) / std > self.cfg.evidence_threshold]
        onset = int(round(over.index[0] - self.anchor)) if len(over) else None
        result = {"metric": metric, "service": r.service, "score": sig(r.score),
                  "baseline": {"mean": sig(r.baseline_mean), "std": sig(r.baseline_std)},
                  "window": {"mean": sig(r.window_mean), "max": sig(r.window_max), "min": sig(r.window_min)},
                  "first_exceeds_threshold_s": onset}
        result["evidence"] = [{"service": r.service, "signal": metric, "kind": "metric", "score": sig(r.score)}]
        return result

    def get_metric_window(self, metric: str, start_s: int | None = None, end_s: int | None = None) -> dict:
        if metric not in self.case.services:
            return {"error": f"unknown metric {metric!r}; list names with top_deviations"}
        start = -self.cfg.baseline_s if start_s is None else start_s
        end = self.cfg.window_s if end_s is None else end_s
        s = self.case.metrics[metric]
        rel = s.index.to_numpy(dtype=float) - self.anchor
        s = s[(rel >= start) & (rel < end)]
        if s.empty:
            return {"metric": metric, "points": [], "evidence": []}
        bins = np.array_split(np.arange(len(s)), min(self.cfg.series_points, len(s)))
        points = [[int(round(s.index[b[0]] - self.anchor)), sig(float(s.iloc[b].mean()))] for b in bins]
        return {"metric": metric, "service": self.case.services[metric], "points_relative_s_value": points,
                "evidence": []}

    # -- logs ------------------------------------------------------------------

    def _log_frame(self) -> pd.DataFrame:
        if self._logs is None:
            logs = self._logs_source()
            logs = logs.assign(phase=self._phase(logs["time"]))
            self._logs = logs[logs["phase"] != ""]
        return self._logs

    def _service_logs(self, service: str) -> pd.DataFrame:
        logs = self._log_frame()
        logs = logs[logs["service"] == service].copy()
        logs["pattern"] = templates(logs["message"])
        level = logs["level"].astype("str") if "level" in logs else pd.Series("", index=logs.index)
        logs["is_error"] = (level.str.contains(r"error|warn|fatal", case=False, regex=True) |
                            logs["message"].str.contains(LEVEL)).fillna(False)
        return logs

    def get_logs(self, service: str) -> dict:
        if "logs" not in self.case.sources:
            return {"error": "this case has no logs"}
        logs = self._service_logs(service)
        ratio = self.cfg.window_s / self.cfg.baseline_s
        counts = logs.groupby(["pattern", "phase"]).size().unstack(fill_value=0)
        for phase in ("baseline", "window"):
            if phase not in counts:
                counts[phase] = 0
        expected = counts["baseline"] * ratio
        counts["score"] = (counts["window"] - expected) / np.sqrt(expected + 1)
        top = counts.reindex(counts["score"].abs().sort_values(ascending=False).index).head(self.cfg.log_patterns)
        errors = logs[logs["is_error"]].groupby("phase").size()
        patterns = [{"pattern": p, "baseline_count": int(r.baseline), "window_count": int(r.window),
                     "score": sig(r.score)} for p, r in top.iterrows()]
        result = {"service": service,
                  "lines": {"baseline": int((logs["phase"] == "baseline").sum()),
                            "window": int((logs["phase"] == "window").sum())},
                  "error_or_warning_lines": {"baseline": int(errors.get("baseline", 0)),
                                             "window": int(errors.get("window", 0))},
                  "window_vs_baseline_length": ratio, "patterns": patterns}
        result["evidence"] = [{"service": service, "signal": p["pattern"], "kind": "log", "score": abs(p["score"])}
                              for p in patterns]
        return result

    # -- traces ----------------------------------------------------------------

    def _trace_frame(self) -> pd.DataFrame:
        if self._traces is None:
            spans = self._traces_source()
            spans = spans.assign(phase=self._phase(spans["start"]))
            spans = spans[spans["phase"] != ""].copy()
            spans["service"] = spans["service"].map(self.trace_service)
            code = spans["status_code"]
            # gRPC: 0 is OK. HTTP: 2xx and 3xx are OK. Train Ticket records no status codes.
            spans["is_error"] = code.notna() & (code != 0) & ((code < 200) | (code >= 400))
            self._traces = spans
        return self._traces

    def get_trace_errors(self, service: str) -> dict:
        if "traces" not in self.case.sources:
            return {"error": "this case has no traces"}
        spans = self._trace_frame()
        spans = spans[spans["service"] == service]
        has_status = bool(self._trace_frame()["status_code"].notna().any())
        operations = []
        for op, g in spans.groupby("operation"):
            base, win = g[g["phase"] == "baseline"], g[g["phase"] == "window"]
            if len(base) < 5 or len(win) < 5:
                continue
            b_med = base["duration_us"].median()
            mad = (base["duration_us"] - b_med).abs().median() * 1.4826
            latency = (win["duration_us"].median() - b_med) / max(mad, 0.05 * b_med, 1.0)
            entry = {"operation": op, "spans": {"baseline": len(base), "window": len(win)},
                     "p50_us": {"baseline": sig(b_med), "window": sig(win["duration_us"].median())},
                     "p95_us": {"baseline": sig(base["duration_us"].quantile(0.95)),
                                "window": sig(win["duration_us"].quantile(0.95))},
                     "latency_score": sig(latency)}
            score = abs(latency)
            if has_status:
                p_b, p_w = base["is_error"].mean(), win["is_error"].mean()
                errors = (p_w - p_b) / math.sqrt(max(p_b * (1 - p_b), 1e-4) / len(win))
                entry["error_rate"] = {"baseline": sig(p_b), "window": sig(p_w)}
                entry["error_score"] = sig(errors)
                score = max(score, abs(errors))
            entry["score"] = sig(score)
            operations.append(entry)
        operations.sort(key=lambda e: -e["score"])
        operations = operations[:self.cfg.log_patterns]
        result = {"service": service, "status_codes_recorded": has_status, "operations": operations}
        result["evidence"] = [{"service": service, "signal": o["operation"], "kind": "trace", "score": o["score"]}
                              for o in operations]
        return result

    def _call_graph(self) -> pd.DataFrame:
        """Cross-service calls (parent span's service -> child span's service), counted once per case."""
        if getattr(self, "_edges", None) is None:
            spans = self._trace_frame()[["trace_id", "span_id", "parent_span_id", "service"]]
            edges = spans.merge(spans, left_on=["trace_id", "parent_span_id"], right_on=["trace_id", "span_id"],
                                suffixes=("", "_parent"))
            edges = edges[edges["service"] != edges["service_parent"]]
            self._edges = (edges.groupby(["service_parent", "service"]).size().rename("calls")
                           .reset_index().sort_values("calls", ascending=False))
        return self._edges

    def call_graph(self) -> list[dict]:
        """Every cross-service call seen in the traces, for display (blast radius); [] without traces."""
        if "traces" not in self.case.sources:
            return []
        return [{"caller": r.service_parent, "callee": r.service, "calls": int(r.calls)}
                for r in self._call_graph().itertuples()]

    def get_dependencies(self, service: str) -> dict:
        if "traces" not in self.case.sources:
            return {"error": "this case has no traces"}
        edges = self._call_graph()
        callers = edges[edges["service"] == service].set_index("service_parent")["calls"]
        callees = edges[edges["service_parent"] == service].set_index("service")["calls"]
        result = {"service": service,
                  "called_by": [{"service": s, "calls": int(n)} for s, n in callers.items()],
                  "calls": [{"service": s, "calls": int(n)} for s, n in callees.items()],
                  "evidence": []}
        if callers.empty and callees.empty:
            result["note"] = ("no cross-service calls recorded for this service; its requests may arrive "
                              "through an untraced gateway")
        return result

    # -- dispatch --------------------------------------------------------------

    TOOLS = ("top_deviations", "compare_to_baseline", "get_metric_window", "get_logs",
             "get_trace_errors", "get_dependencies")

    def call(self, name: str, args: dict) -> ToolCall:
        """Run one tool and record it. Errors come back as results so an agent can recover."""
        call_id = f"c{len(self.calls) + 1}"
        if name not in self.TOOLS:
            result = {"error": f"unknown tool {name!r}"}
        elif "service" in args and args["service"] is not None and args["service"] not in self.case.candidate_services:
            result = {"error": f"unknown service {args['service']!r}"}
        else:
            tool = getattr(self, name)
            try:
                inspect.signature(tool).bind(**args)
            except TypeError as error:
                result = {"error": f"bad arguments for {name}: {error}"}
            else:
                result = tool(**args)
        evidence = result.pop("evidence", [])
        record = ToolCall(call_id, name, dict(args), result, evidence)
        self.calls.append(record)
        return record

    def output(self, record: ToolCall) -> str:
        """The JSON a model sees for one call, including its id for citations."""
        return json.dumps({"call_id": record.call_id, **record.result}, separators=(",", ":"))
