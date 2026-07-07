"""
Script to analyze and optimize imports in spectral_analyzer.py
This helps identify unused imports that can be removed to reduce executable size.

@author: Sharad Kumar Gupta
"""

import ast
import sys
from pathlib import Path

def analyze_imports(file_path):
    """Analyze imports in the given Python file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
    
    return imports

def find_unused_imports(file_path):
    """Find potentially unused imports"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports = analyze_imports(file_path)
    unused = []
    
    for imp in imports:
        # Simple check - look for usage in the file
        # This is a basic heuristic and may have false positives
        base_name = imp.split('.')[-1]
        if base_name not in content.replace(f"import {imp}", "").replace(f"from {imp}", ""):
            unused.append(imp)
    
    return unused

def main():
    file_path = Path("spectral_analyzer.py")
    if not file_path.exists():
        print("spectral_analyzer.py not found!")
        return
    
    print("Analyzing imports in spectral_analyzer.py...")
    imports = analyze_imports(file_path)
    
    print(f"\nFound {len(imports)} imports:")
    for imp in sorted(imports):
        print(f"  - {imp}")
    
    print("\nPotentially unused imports (manual verification needed):")
    unused = find_unused_imports(file_path)
    for imp in unused:
        print(f"  - {imp}")
    
    if not unused:
        print("  None found - all imports appear to be used")

if __name__ == "__main__":
    main()