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
        """True if *step* satisfies the rule's condition."""
        # Minimal evaluator — real implementation uses a safe expr parser
        env = step.as_dict()
        try:
            # condition examples:  "tool_name == 'shell'", "risk_score > 0.8"
            return bool(eval(self.condition, {"__builtins__": {}}, env))  # noqa: S307
        except Exception:
            return True  # fail-open during evaluation; log separately


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