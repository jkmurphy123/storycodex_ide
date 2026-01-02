# Project Context

This editor is part of a larger StoryCodex pipeline.

StoryCodex:
- Generates scene draft markdown files
- Expects clean markdown output
- Treats paragraphs as semantic units

Design principles:
- Paragraph is the atomic edit unit
- AI assists, never auto-overwrites silently
- Human remains in control at all times

Future plans:
- Paragraph diff preview
- Rewrite history browser
- Style presets

Non-goals:
- Full document AI rewriting
- Chat-style editing
- Web-based UI
