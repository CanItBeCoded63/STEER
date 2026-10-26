# STEER: Specification-Driven Agent Steering and Verification

**STEER** is a framework for formally specifying, verifying, and governing LLM agent behavior at scale.

## Why STEER?

LLM agents are deployed without verifiable guarantees. Steering files (`CLAUDE.md`, `AGENTS.md`) are informal and cannot catch drift, hallucination, or unsafe behavior at scale. STEER provides:

- **Typed specification language** for agent contracts
- **Runtime trajectory verification** that catches violations in real time
- **Multi-agent collaboration protocols** with signed commitments
- **No-code spec authoring** via Markdown rendering
- **Native integration** with AWS Bedrock, SageMaker, and Helix DB

## Quickstart

```bash
pip install steer-ai
```

```python
from steer_ai.spec import AgentSpec, Rule, Severity, RuleType, ActionType, TrajectoryStep
from steer_ai.verify import TrajectoryVerifier
from pathlib import Path

# Define a spec
spec = AgentSpec(
    spec_id="coding-agent-v1",
    agent_name="CodeAgent",
    description="A coding agent that runs safely in a sandbox",
    rules=[
        Rule(
            rule_type=RuleType.DENY_LIST,
            action=ActionType.TOOL_CALL,
            condition="tool_name == 'shell' and 'sudo' in tool_args.get('command', '')",
            severity=Severity.BLOCK,
            description="Agent must not execute sudo commands",
        ),
    ],
    safety_guards=[
        {
            "invariant": "never disclose secrets",
            "severity": Severity.BLOCK,
        },
    ],
)

# Verify a trajectory
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
]

verifier = TrajectoryVerifier(spec)
report = verifier.verify_trajectory(traj)

print(f"Status: {report.status}")
print(f"Violations: {report.violation_count}")
```

## Specification Language (SSL)

STEER uses a typed DSL (Specification Language for LLM agents) that can be:

- **Written as Markdown** (easily maintained alongside code)
- **Serialized as JSON** (for API / configuration)
- **Rendered to/from Markdown** (for review and migration)

Key constructs:

| Construct | Description |
|-----------|-------------|
| `Rule` | Machine-checkable constraint on an action |
| `ALLOW_LIST` | Permit only specific tools/data |
| `DENY_LIST` | Prohibit specific patterns |
| `SAFETY_GUARD` | Invariants like "never disclose PII" |
| `COLLABORATION` | Multi-agent interaction contracts |

See `src/steer_ai/spec.py` for the full language.

## Graph Memory Integration

STEER optionally integrates with graph databases like **Helix DB** for persistent agent memory. The spec can reference a `memory_graph_uri`:

```python
spec = AgentSpec(
    ...,
    memory_graph_uri="helix://graph/agent-experience-v1",
    collaboration=[
        {"peer_role": "peer-analyst", "allowed_interaction": "observe"},
    ],
)
```

This enables agents to reference shared knowledge while staying within verified bounds.

## Cloud Integration

STEER is designed to be embedded in AWS workflows:

- **Bedrock Agent Runtime**: Verification middleware between agent and tool execution
- **Lambda**: Stateless verification service for serverless agent flows  
- **SageMaker**: Training verification models on labeled trajectory data
- **OpenSearch**: Vector search for spec retrieval and policy matching

## License

MIT — feel free to use, modify, and contribute.