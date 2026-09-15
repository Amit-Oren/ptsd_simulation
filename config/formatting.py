"""
formatting.py — renders domain data (demographics, self-report, validator
issues) into prompt-ready text. Pure functions, no state — lives next to
config/demographics.py and config/self_report.py, which define the shapes
of the data these functions render, rather than on any agent class (they
never depended on agent state, just happened to be defined there).
"""


def fmt_demographics(demographics: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in demographics.items())


def fmt_self_report(self_report: dict) -> str:
    lines = []
    for node, items in self_report.items():
        formatted = ", ".join(
            f"{i['key']}: {i['value']}" if isinstance(i, dict) else str(i)
            for i in items
        )
        lines.append(f"  {node}: {formatted}")
    return "\n".join(lines)


def fmt_issues(issues: list[dict]) -> str:
    lines = []
    for issue in issues:
        lines.append(f"  {issue['component']} / {issue['item']}: {issue['explanation']}")
    return "\n".join(lines) or "  (none)"
