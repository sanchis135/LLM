---
id: mlops-playbook
title: Open MLOps Playbook (Concise)
license: CC BY 4.0 (© 2025 Sandra Martínez Sanchis)
version: 1.0
---


# Model Registry & Versioning
- Use semantic versioning (MAJOR.MINOR.PATCH) for models.
- Store artifacts (model + preprocessing) with immutable digests.
- Record training data snapshot and feature schema.


# CI/CD for Models
- Stages: lint → unit tests → data checks → train → evaluate → package → push → deploy.
- Block promotion if performance or fairness gates fail.
- Use environment‑specific configs (dev/stage/prod) and IaC for infra.


# Deployment Strategies
- Canary: route 5–10% traffic to the new version, monitor SLOs, then ramp.
- Blue‑green: keep blue live, deploy green, switch when healthy.
- Shadow: mirror traffic to candidate model without affecting users.


# Monitoring & Observability
- Track: latency (p50/p95), error rate, throughput, CPU/GPU/mem.
- Model performance: accuracy/MAE, concept drift, data drift, feature skew.
- Log prediction inputs/outputs with PII redaction.


# Risk & Governance
- Document model cards, approvals, and rollback plans.
- Keep audit trails (who deployed what, when, and why).