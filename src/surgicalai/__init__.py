"""SurgicalAI - AI-powered surgical planning for dermatologic reconstruction."""

__version__ = "0.1.0"

# Warn if Python version is >=3.12
import os, sys
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
check_script = os.path.join(repo_root, "python_version_check.py")
if os.path.exists(check_script):
	try:
		exec(open(check_script).read())
	except Exception:
		pass
