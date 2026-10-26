"""Runtime trajectory verification engine.

Given an AgentSpec and a stream of TrajectorySteps (a full agent session),
STEER's verifier checks each action against the spec's rules and reports
violations, warnings, and audit entries.  Designed to be embeddable in
Bedrock agent flows, Lambda verification services, or multi-agent graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from .spec import AgentSpec, Rule, RuleType, Severity, TrajectoryStep, ActionType


class Verdict(str, Enum):
    """Result of a single verification check."""
    PASS = "pass"
    VIOLATION = "violation"
    WARNING = "warning"
    AUDITED = "audited"


@dataclass
class VerificationResult:
    """Outcome of verifying one step against the spec."""
    step_idx: int
    verdict: Verdict
    rule_id: int | None
    message: str
    severity: Severity | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Aggregated results for a complete trajectory."""
    spec_id: str
    total_steps: int
    violations: list[VerificationResult] = field(default_factory=list)
    warnings: list[VerificationResult] = field(default_factory=list)
    audited: list[VerificationResult] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.violations:
            return "failed"
        return "passed"

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "total_steps": self.total_steps,
            "status": self.status,
            "violations": [v.__dict__ for v in self.violations],
            "warnings": [w.__dict__ for w in self.warnings],
            "audited_count": len(self.audited),
        }


class TrajectoryVerifier:
    """Checks agent trajectories against an AgentSpec at runtime.

    Usage::

        verifier = TrajectoryVerifier.load_spec("coding-agent.json")
        report = verifier.verify_trajectory(trajectory_steps)
        report.violation_count   # → how many violations
    """

    def __init__(self, spec: AgentSpec):
        self.spec = spec

    @classmethod
    def load_spec(cls, path: str) -> "TrajectoryVerifier":
        return cls(AgentSpec.load(path))

    def verify_step(self, step: TrajectoryStep) -> list[VerificationResult]:
        """Check a single step against all applicable rules."""
        results: list[VerificationResult] = []

        # --- Generic rules ---
        for idx, rule in enumerate(self.spec.rules):
            if rule.action and rule.action != step.action_type:
                continue
            if not rule.matches(step):
                # Build a meaningful message
                msg = rule.description or f"Rule violation at step {step.step_idx}"
                verdict = {
                    Severity.BLOCK: Verdict.VIOLATION,
                    Severity.WARN: Verdict.WARNING,
                    Severity.AUDIT: Verdict.AUDITED,
                }[rule.severity]

                results.append(VerificationResult(
                    step_idx=step.step_idx,
                    verdict=verdict,
                    rule_id=idx,
                    message=msg,
                    severity=rule.severity,
                    details={"condition": rule.condition, "step": step.as_dict()},
                ))

        # --- Safety guards ---
        for sg in self.spec.safety_guards:
            # Simple keyword-based guard for now
            combined_text = " ".join(
                str(v) for v in {**step.tool_args, **step.metadata}.values()
            )
            if sg.invariant.split()[0].lower() not in combined_text.lower():
                # Guard passed for this step
                continue
            results.append(VerificationResult(
                step_idx=step.step_idx,
                verdict=Verdict.VIOLATION if sg.severity == Severity.BLOCK
                           else Verdict.WARNING,
                rule_id=None,
                message=f"Safety guard triggered: {sg.invariant}",
                severity=sg.severity,
                details={"step": step.as_dict()},
            ))

        # Default pass if nothing triggered
        if not results:
            results.append(VerificationResult(
                step_idx=step.step_idx,
                verdict=Verdict.PASS,
                rule_id=None,
                message="All checks passed",
            ))

        return results

    def verify_trajectory(
        self, trajectory: list[TrajectoryStep] | Iterator[TrajectoryStep]
    ) -> VerificationReport:
        """Verify a full agent trajectory and return a report."""
        results = VerificationReport(
            spec_id=self.spec.spec_id,
            total_steps=0,
        )

        # If it's an iterator, consume it
        steps = list(trajectory) if not isinstance(trajectory, list) else trajectory

        results.total_steps = len(steps)

        for step in steps:
            step_results = self.verify_step(step)
            for r in step_results:
                if r.verdict == Verdict.VIOLATION:
                    results.violations.append(r)
                elif r.verdict == Verdict.WARNING:
                    results.warnings.append(r)
                elif r.verdict == Verdict.AUDITED:
                    results.audited.append(r)

        return results


# --- Example usage ---
if __name__ == "__main__":
    import json

    # Load a spec
    spec = AgentSpec(
        spec_id="demo-coding-v1",
        agent_name="CodeAgent",
        description="A coding agent that must not run sudo or access /etc/passwd",
        rules=[
            Rule(
                rule_type=RuleType.DENY_LIST,
                action=ActionType.TOOL_CALL,
                condition="tool_name == 'shell' and 'sudo' in tool_args.get('command', '')",
                severity=Severity.BLOCK,
                description="Agent must not execute sudo commands",
            ),
            Rule(
                rule_type=RuleType.DENY_LIST,
                action=ActionType.FILE_WRITE,
                condition="tool_args.get('path', '').startswith('/etc/')",
                severity=Severity.BLOCK,
                description="Agent must not write to system files",
            ),
        ],
        safety_guards=[
            {"invariant": "never disclose secrets", "severity": Severity.BLOCK}
        ],
    )

    # Simulate a trajectory
    traj: list[TrajectoryStep] = [
        TrajectoryStep(
            step_idx=0,
            action_type=ActionType.TOOL_CALL,
            tool_name="shell",
            tool_args={"command": "ls -la"},
            risk_score=0.1,
        ),
        TrajectoryStep(
            step_idx=1,
            action_type=ActionType.TOOL_CALL,
            tool_name="shell",
            tool_args={"command": "sudo apt-get update"},
            risk_score=0.95,
        ),
        TrajectoryStep(
            step_idx=2,
            action_type=ActionType.FILE_WRITE,
            tool_args={"path": "src/app.py", "content": "print('hello')"},
            risk_score=0.8,
        ),
    ]

    verifier = TrajectoryVerifier(spec)
    report = verifier.verify_trajectory(traj)

    print(json.dumps(report.to_dict(), indent=2))