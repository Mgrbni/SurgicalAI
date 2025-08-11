# CHANGES

## Demo Pipeline & Contract Unification (Date: 2025-08-11)

Implemented end-to-end photo → /api/analyze → artifacts (heatmap, overlay, plan, metrics, PDF) → UI rendering path.

Key updates:

1. API Contract
   - /api/analyze now returns both legacy keys and new unified contract fields:
     * run_id (timestamp_uuid4short)
     * diagnosis (subset of probabilities)
     * flap (suggestion object)
     * artifacts_list (array of {name,path} with POSIX paths under runs/<run_id>/... )
   - Added structured error responses {"error": {code,message}} for common failure cases.
   - Added /api/health returning {"ok": true} (legacy /health & /healthz preserved).
   - All artifacts stored at runs/<run_id>/ (removed implicit /demo/ subdir for new flow).
   - Added plan.json and run.json (manifest) per run.

2. Artifact Serving
   - /api/artifact/{path} now expects paths like runs/<run_id>/file.ext and enforces safe traversal checks with logging (200 vs 404).
   - Added INFO log lines for each access: "Artifact 200" / "Artifact 404".

3. Report Generation
   - build_report invoked with produce_pdf=True to ensure report.pdf is emitted; fallback already handled.
   - PDF + HTML paths included in artifacts_list.

4. Flap Suggestion
   - Simple heuristic flap suggestion included (rotation vs advancement) based on gate + location.
   - plan.json persisted with flap, gate, run metadata.

5. Client (client/api.js)
   - Adapted to prefer artifacts_list; gracefully falls back to legacy mapping.
   - Renders probabilities from diagnosis|probs|lesion_probs.
   - Normalized percentage display even if probabilities already in 0-1 or 0-100.
   - "Open Report" button now opens PDF (report artifact) reliably.
   - Robust error toasts improved.

6. Environment / Config
   - Cleaned .env.example with commentary and clarified key usage.

7. Backward Compatibility
   - Legacy response keys (probs, lesion_probs, artifacts.overlay_png etc.) retained so older UI/test code continues to function.
   - /health and /healthz remain available.

8. Logging
   - Unified logger name "surgicalai"; improved warnings for traversal attempts.

Future considerations:
   - Remove deprecated legacy keys after frontend fully migrated.
   - Add test_demo.py (pending) for new contract validation including PDF existence.

