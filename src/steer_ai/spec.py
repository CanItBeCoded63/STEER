"""STEER Specification Language (SSL) — machine-checkable agent contracts.

A typed DSL describing agent behavior constraints: what tools an agent
may use, what data it may access, what actions are forbidden, and what
safety policies it must follow.  Specs can be imported/exported as JSON,
rendered to/from Markdown for human review, and compiled into fast
runtime checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Categories of actions an agent may perform."""
    TOOL_CALL = "tool_call"
    DATA_ACCESS = "data_access"
    EXTERNAL_API = "external_api"
    FILE_WRITE = "file_write"
    MEETING_ACTION = "meeting_action"
    DELEGATION = "delegation"


class SafetyTier(str, Enum):
    """Governance tiers for regulated / high-stakes operation."""
    RESTRICTED = "restricted"   # strict allow-list only
    STANDARD = "standard"       # default production
    EXPERIMENT = "experiment"   # research / dev


class Severity(str, Enum):
    BLOCK = "block"      # prevent action
    WARN = "warn"        # log + surface to human
    AUDIT = "audit"      # log for later review


class RuleType(str, Enum):
    """Kinds of constraint rules in the spec language."""
    ALLOW_LIST = "allow_list"
    DENY_LIST = "deny_list"
    CONSTRAINT = "constraint"
    SAFETY_GUARD = "safety_guard"
    COLLABORATION = "collaboration"


# --- Safe expression evaluator (no eval/exec) -------------------------------

import ast
import operator


_ALLOWED_BINOPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.And: operator.and_,
    ast.Or: operator.or_,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_ALLOWED_UNARYOPS: dict[type, object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_ALLOWED_CALLABLES: dict[str, object] = {
    "str.startswith": lambda s, p: s.startswith(p),
    "str.endswith": lambda s, p: s.endswith(p),
    "str.lower": lambda s: s.lower(),
    "str.upper": lambda s: s.upper(),
    "str.contains": lambda s, p: p in s,
    "dict.get": lambda d, k, default=None: d.get(k, default),
}


def _safe_eval(expr: str, env: dict[str, Any]) -> Any:
    """Evaluate a whitelisted expression against *env*.

    Only a restricted subset of Python expressions is permitted:
    arithmetic, comparisons, boolean logic, membership tests, and
    attribute access on a small set of safe callables.  Everything
    else raises ``TypeError``.
    """
    tree = ast.parse(expr.strip(), mode="eval")
    return _eval_node(tree.body, env)


def _eval_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.value, env)
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        return any(values)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise TypeError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(operand)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, env)
            cmp_op = _ALLOWED_BINOPS.get(type(op))
            if cmp_op is None:
                raise TypeError(f"Unsupported comparison: {type(op).__name__}")
            if not cmp_op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        func = _eval_node(node.func, env)
        args = [_eval_node(a, env) for a in node.args]
        if not callable(func):
            raise TypeError("Called object is not callable")
        return func(*args)
    if isinstance(node, ast.Attribute):
        value = _eval_node(node.value, env)
        # Only allow attribute access on safe callables
        if isinstance(value, type):
            name = f"{value.__name__}.{node.attr}"
            if name in _ALLOWED_CALLABLES:
                return _ALLOWED_CALLABLES[name]
            raise TypeError(f"Attribute access not allowed: {name}")
        if isinstance(value, dict):
            return value.get(node.attr)
        raise TypeError("Attribute access only allowed on dicts")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise NameError(f"Name '{node.id}' is not defined")
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_eval_node(e, env) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(e, env) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval_node(k, env): _eval_node(v, env) for k, v in zip(node.keys, node.values)}
    raise TypeError(f"Unsupported expression node: {type(node).__name__}")


@dataclass(frozen=True)
class Rule:
    """A single constraint in an agent's behavior contract."""
    rule_type: RuleType
    action: ActionType | None
    condition: str                         # dot-notation expression over a TrajectoryStep
    severity: Severity = Severity.WARN
    description: str = ""
    allowed_values: list[str] | None = None

    def matches(self, step: "TrajectoryStep") -> bool:
        """True if *step* satisfies the rule's condition.

        Uses a sandboxed AST-based evaluator (no ``eval``/``exec``) that
        whitelists a safe subset of Python expressions: comparisons,
        boolean logic, membership tests, and a small set of safe
        callables (``str.startswith``, ``str.endswith``, ``str.lower``,
        ``str.upper``, ``str.contains``, ``dict.get``).
        """
        env = step.as_dict()
        try:
            return bool(_safe_eval(self.condition, env))
        except Exception:
            # Fail-open during evaluation; log separately
            return True


class ToolAllowRule(BaseModel):
    """Allow-list rule for tool invocations."""
    tools: list[str] = Field(default_factory=list)
    reason: str | None = None


class DataAccessRule(BaseModel):
    """Rule governing what data an agent may read/write."""
    namespaces: list[str] = Field(default_factory=list)
    read_only: bool = True


class SafetyGuard(BaseModel):
    """A safety guardrail — e.g. 'never disclose PII'."""
    invariant: str
    severity: Severity = Severity.BLOCK
    remediation: str | None = None


class CollaborationRule(BaseModel):
    """Contract element governing multi-agent interaction."""
    peer_role: str
    allowed_interaction: str = "request_response"   # or observe, delegate
    required_signoff: bool = False


class AgentSpec(BaseModel):
    """A full STEER specification for one agent (or agent class).

    Think of this as a formally-checkable superset of a `CLAUDE.md` or
    `AGENTS.md` steering file: structured, typed, and verifiable.
    """
    spec_id: str = Field(..., description="Unique identifier, e.g. 'team-a-coding-v1'")
    version: str = "1.0.0"
    agent_name: str
    description: str = ""
    tier: SafetyTier = SafetyTier.STANDARD

    # --- Behavior constraints ---
    tools: list[ToolAllowRule] = Field(default_factory=list)
    data: list[DataAccessRule] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)   # generic conditions
    safety_guards: list[SafetyGuard] = Field(default_factory=list)

    # --- Multi-agent contracts ---
    collaboration: list[CollaborationRule] = Field(default_factory=list)

    # --- Optional graph-memory references ---
    memory_graph_uri: str | None = None  # e.g. "helix://graph/team-memory"

    # --- Round-trip serialization ---
    def to_markdown(self) -> str:
        """Render spec as human-readable Markdown (for review / CLAUDE.md migration)."""
        lines = [f"# STEER Spec: {self.agent_name}", "", f"*{self.description}*", "",
                 f"Tier: {self.tier.value}  |  Spec ID: {self.spec_id}", ""]
        if self.tools:
            lines += ["## Tool Allow-list", ""]
            for tr in self.tools:
                lines.append(f"- `{', '.join(tr.tools)}`" + (f" — {tr.reason}" if tr.reason else ""))
        if self.safety_guards:
            lines += ["", "## Safety Guards", ""]
            for sg in self.safety_guards:
                lines.append(f"- [{sg.severity.value}] {sg.invariant}" +
                             (f" → {sg.remediation}" if sg.remediation else ""))
        if self.collaboration:
            lines += ["", "## Multi-Agent Collaboration", ""]
            for c in self.collaboration:
                lines.append(f"- Peer `{c.peer_role}`: {c.allowed_interaction}" +
                             (" (signoff required)" if c.required_signoff else ""))
        return "\n".join(lines) + "\n"

    @classmethod
    def from_markdown(cls, text: str) -> "AgentSpec":
        """Parse a Markdown steering file into a structured spec (best-effort)."""
        # Minimal parser for the reverse direction
        raise NotImplementedError("Markdown → spec parser (MVP)")

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AgentSpec":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        # Reconstruct Rule objects from stored dicts
        spec_dict = dict(data)
        spec_dict["rules"] = [Rule(**r) if isinstance(r, dict) else r for r in data.get("rules", [])]
        return cls(**spec_dict)


@dataclass
class TrajectoryStep:
    """One step in an agent's execution log."""
    step_idx: int
    action_type: ActionType
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    data_access: str | None = None
    risk_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = self.metadata | {
            "step_idx": self.step_idx,
            "action_type": self.action_type.value,
            "tool_name": self.tool_name,
            "risk_score": self.risk_score,
        }
        if self.tool_args:
            d.update(self.tool_args)
        if self.data_access:
            d["data_access"] = self.data_access
        return d


# JSON import for load()
import json  # noqa: E402