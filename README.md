# Scene Paragraph Editor

A desktop GUI for editing markdown scene files produced by StoryCodex.

Features:
- Paragraph-by-paragraph editing
- AI-assisted rewrites via OpenAI or Ollama
- Paragraph-scoped undo (Ctrl+Z)
- Safe markdown save

## Requirements

- Python 3.10+
- PySide6
- requests
- openai (optional, for OpenAI backend)

## Run

```bash
pip install -r requirements.txt
python scene_editor.py
