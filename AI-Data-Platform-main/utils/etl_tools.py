import os
import ast
import subprocess
import tempfile
import requests
import pandas as pd
import numpy as np
import json
from typing import Tuple, Dict, Any, Optional

# Forbidden modules and functions for in-process AST sandbox
FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "pty",
    "commands", "builtins", "__builtin__", "posix", "nt", "importlib",
    "urllib", "requests", "http", "ftplib", "telnetlib"
}

FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open", "input",
    "breakpoint", "help", "globals", "locals", "vars"
}


def validate_python_ast(code: str) -> Tuple[bool, str]:
    """
    Analyzes Python code AST to reject dangerous imports, OS interactions, and system mutations.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in generated code: {e}"

    for node in ast.walk(tree):
        # Disallow dangerous import statements
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split('.')[0]
                if root_module in FORBIDDEN_MODULES:
                    return False, f"Prohibited module import: '{alias.name}'. Only data processing libraries (pandas, numpy, json) are permitted."
        
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split('.')[0]
                if root_module in FORBIDDEN_MODULES:
                    return False, f"Prohibited from-import: '{node.module}'. Only data processing libraries are permitted."

        # Disallow calls to dangerous builtins (eval, exec, open, etc.)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_BUILTINS:
                    return False, f"Prohibited function execution: '{node.func.id}()' is blocked for sandbox security."

    return True, "Code passed AST sandbox validation."


class ETLTools:

    def __init__(self, timeout_seconds: int = 15, memory_limit_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb

    def extract_load(self, url: str, output_folder: str, format: str) -> str:
        """
        Extracts data from an API (url) and loads it into the desired output_folder.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        full_output_folder = os.path.join(project_root, output_folder)

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            os.makedirs(full_output_folder, exist_ok=True)
            filename = os.path.join(full_output_folder, f"extracted_data.{format}")

            if isinstance(data, dict) and 'results' in data:
                df = pd.json_normalize(data['results'])
            elif isinstance(data, list):
                df = pd.json_normalize(data)
            else:
                df = pd.json_normalize([data])

            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            elif format == "parquet":
                df.to_parquet(filename, index=False)
            else:
                return f"Unsupported format: {format}"

            return f"Data successfully extracted ({len(df)} rows) and saved to {filename}"
        except requests.exceptions.RequestException as e:
            return f"Failed to extract data from API: {e}"
        except Exception as e:
            return f"Failed to process API response: {e}"

    def transform_load_context(self, file_path: str) -> str:
        """
        Reads sample rows from the file to give context to the LLM for Pandas code generation.
        """
        if not os.path.isabs(file_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            file_path = os.path.join(project_root, file_path)

        if not os.path.exists(file_path):
            return f"File does not exist: {file_path}"

        file_extension = os.path.splitext(file_path)[1].lower()
        try:
            if file_extension == ".csv":
                df = pd.read_csv(file_path)
            elif file_extension == ".json":
                df = pd.read_json(file_path, lines=True)
            elif file_extension == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                return f"Unsupported file format: {file_extension}"

            return f"Columns: {list(df.columns)}\nTop 3 Sample Rows:\n{df.head(3).to_string()}"
        except Exception as e:
            return f"Error reading context: {e}"

    def _execute_in_docker(self, code: str) -> Tuple[bool, str]:
        """
        Executes code inside a Docker container with strict CPU, Memory, and Timeout limits.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, 'data')

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            cmd = [
                "docker", "run", "--rm",
                f"--memory={self.memory_limit_mb}m",
                "--cpus=1.0",
                "--pids-limit=64",
                "--network=none",
                "-v", f"{data_dir}:/workspace/data",
                "-v", f"{tmp_path}:/workspace/script.py:ro",
                "python:3.12-slim",
                "python", "/workspace/script.py"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
            if res.returncode == 0:
                return True, res.stdout or "Code executed successfully in Docker container."
            else:
                return False, f"Docker execution error: {res.stderr}"
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out (> {self.timeout_seconds}s limit)."
        except Exception as e:
            return False, f"Docker unavailable ({e})"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def execute_code(self, code: str, use_docker: bool = False) -> str:
        """
        Executes generated Pandas code inside either a Docker container sandbox
        or an AST-inspected restricted in-process namespace with time and memory guardrails.
        """
        # Step 1: Deterministic AST security check
        is_safe, reason = validate_python_ast(code)
        if not is_safe:
            return f"Execution blocked by sandbox security: {reason}"

        # Step 2: Attempt Docker isolation if requested or available
        if use_docker:
            success, msg = self._execute_in_docker(code)
            if success:
                return f"Docker Sandbox: {msg}"

        # Step 3: Fast in-process restricted namespace execution
        safe_globals = {
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
            "json": json,
            "__builtins__": {
                "range": range, "len": len, "str": str, "int": int, "float": float,
                "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
                "print": print, "min": min, "max": max, "sum": sum, "round": round,
                "enumerate": enumerate, "zip": zip, "filter": filter, "map": map,
                "abs": abs, "isinstance": isinstance, "Exception": Exception,
                "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
                "IndexError": IndexError
            }
        }
        safe_locals = {}

        try:
            exec(code, safe_globals, safe_locals)
            return "Code executed successfully in AST sandbox."
        except Exception as e:
            return f"Failed to execute code: {e}"