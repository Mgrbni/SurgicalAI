"""Expose generate_summary for src-layout imports."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../server')))
try:
    from ai_openai import summarize_case as generate_summary
except ImportError:
    def generate_summary(*args, **kwargs):
        return "Summary unavailable (ai_openai import failed)"
