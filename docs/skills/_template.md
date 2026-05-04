# <Skill Name>

> **One-line summary:** <what this skill computes / decides>
> **Source of truth:** [`<relative/path.py>:<LINE-LINE>`](../../<relative/path.py>)
> **Phase:** 1 | 2  •  **Group:** strategy / regime / bias / data_quality / risk / architecture
> **Depends on:** `NN_other_skill.md`, …
> **Consumed by:** `<module.py>` — <one-sentence purpose>

---

## 1. Theory & Objective

What problem does this solve? What were the alternatives? Why this approach?

Keep this section _tight_ — 3–6 sentences. The math goes in §2, the code in §3.

## 2. Mathematical Formula

Use LaTeX-ish notation in fenced blocks. Define every variable, give units, give expected ranges.

```text
score = f(x, y, z)

where
  x ∈ [0, 1]   — <description>
  y ∈ ℝ⁺      — <description>
  z ∈ {0, 1}   — <description>
```

If there's no math, write **"N/A — pure data structure"** or **"N/A — control flow only"** and explain why.

## 3. Reference Python Implementation

Self-contained, copy-pastable. Quote the **actual source verbatim** from the file linked above. Do not paraphrase.

```python
# <relative/path.py>:<LINE-LINE>
def the_function(...):
    ...
```

If the live source spans multiple call sites (e.g. a constant defined in one file, a consumer in another), include each block separately under its own `path:line` header.

## 4. Edge Cases / Guardrails

Bullet list. Each bullet states the failure mode and how upstream code handles it.

- **<Failure mode 1>** — <how it manifests, how it's caught/recovered>
- **<Failure mode 2>** — <…>
- **Sentinel returns** — `None` means …, `0.0` means …, etc.

## 5. Cross-References

- `NN_related_skill.md` — <one-line relationship>
- `NN_downstream_skill.md` — <one-line relationship>

---

*Last verified against repo HEAD on YYYY-MM-DD.*
