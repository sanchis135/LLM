---
id: kubernetes-ops-cheatsheet
title: Kubernetes Ops Cheatsheet (Concise)
license: CC BY 4.0 (© 2025 Sandra Martínez Sanchis)
version: 1.0
---


## Probes & Health
- Liveness detects deadlocks; Readiness gates traffic; Startup for slow apps.
- Example YAML snippet fields: `httpGet`, `initialDelaySeconds`, `periodSeconds`.


## Rollouts
- Use `maxSurge` and `maxUnavailable` to control rolling updates.
- For canary, combine Deployment with Service + `% traffic` via Ingress/ServiceMesh.


## Resources
- Requests/limits per container; tune to avoid CPU throttling and OOMKills.


## Debugging
- `kubectl describe`, `kubectl logs -f`, `kubectl exec -it`.
- Events often reveal scheduling and probe failures.

## OpenShift Weighted Route (90/10)
```yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: my-model
spec:
  to:
    kind: Service
    name: my-model-stable
    weight: 90
  alternateBackends:
    - kind: Service
      name: my-model-canary
      weight: 10
  port:
    targetPort: 8080
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: my-model
spec:
  hosts: ["my-model.apps.example.com"]
  http:
  - route:
    - destination: { host: my-model, subset: stable }
      weight: 90
    - destination: { host: my-model, subset: canary }
      weight: 10
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: my-model
spec:
  host: my-model
  subsets:
  - name: stable
    labels: { version: stable }
  - name: canary
    labels: { version: canary }
```