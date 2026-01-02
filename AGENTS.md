# Scene Paragraph Editor – Codex Instructions

This repository contains a PySide6 desktop application for editing
markdown scene files produced by StoryCodex.

Codex is authorized to:
- Modify existing Python files
- Add new Python files if necessary
- Fix bugs
- Add small, isolated features

Codex must NOT:
- Rewrite the entire application
- Change UI frameworks
- Remove OpenAI or Ollama support
- Introduce new dependencies without explicit instruction

## Architecture Overview

- scene_editor.py
  - Main PySide6 application
  - Paragraph-based markdown editor
  - AI-assisted rewrite per paragraph
  - Ctrl+Z undo is paragraph-scoped

- config.json
  - Selects LLM provider (openai or ollama)
  - Contains prompt templates

## Coding Rules

- Keep changes minimal and localized
- Preserve existing behavior unless explicitly asked
- Prefer small helper methods over large refactors
- Use standard library when possible

## Testing

After making changes, Codex should:
1. Run the application
2. Verify that:
   - File open/save still works
   - Paragraph selection still works
   - Ctrl+Z undo still works
   - AI rewrite still returns plain text

If unsure, explain assumptions in comments.
