import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional
from collections import deque, defaultdict

import requests
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Markdown paragraph handling
# ============================================================

def split_paragraphs(md_text: str) -> List[str]:
    text = md_text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n+", text.strip("\n"))
    return parts if parts else [""]


def join_paragraphs(paragraphs: List[str]) -> str:
    return "\n\n".join(p.rstrip() for p in paragraphs).rstrip() + "\n"


def paragraph_char_ranges(paragraphs: List[str]) -> List[tuple[int, int]]:
    ranges = []
    pos = 0
    for i, p in enumerate(paragraphs):
        start = pos
        end = pos + len(p)
        ranges.append((start, end))
        pos = end
        pos += 2 if i < len(paragraphs) - 1 else 1
    return ranges


# ============================================================
# LLM abstraction
# ============================================================

@dataclass
class LLMConfig:
    provider: str
    prompt_template: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    openai_model: str = "gpt-4.1"
    openai_api_key_env: str = "OPENAI_API_KEY"


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = None

        if cfg.provider == "openai":
            if OpenAI is None:
                raise RuntimeError("openai package missing")
            key = os.getenv(cfg.openai_api_key_env)
            if not key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=key)

    def rewrite(self, instruction: str, paragraph: str) -> str:
        prompt = self.cfg.prompt_template.format(
            instruction=instruction.strip(),
            paragraph=paragraph
        )

        if self.cfg.provider == "ollama":
            return self._rewrite_ollama(prompt)
        if self.cfg.provider == "openai":
            return self._rewrite_openai(prompt)

        raise ValueError("Unknown LLM provider")

    def _rewrite_ollama(self, prompt: str) -> str:
        r = requests.post(
            f"{self.cfg.ollama_base_url}/api/generate",
            json={"model": self.cfg.ollama_model, "prompt": prompt, "stream": False},
            timeout=120
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _rewrite_openai(self, prompt: str) -> str:
        resp = self.client.responses.create(
            model=self.cfg.openai_model,
            input=prompt
        )

        # Preferred SDK helper
        try:
            text = resp.output_text()
            if text:
                return text.strip()
        except Exception:
            pass

        # Manual extraction fallback
        chunks = []
        for msg in getattr(resp, "output", []):
            for item in getattr(msg, "content", []):
                if getattr(item, "type", None) == "output_text":
                    chunks.append(item.text)

        if chunks:
            return "\n".join(chunks).strip()

        raise RuntimeError("OpenAI response contained no output text")


class RewriteWorker(QtCore.QObject):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, llm: LLMClient, instruction: str, paragraph: str):
        super().__init__()
        self.llm = llm
        self.instruction = instruction
        self.paragraph = paragraph

    @QtCore.Slot()
    def run(self):
        try:
            text = self.llm.rewrite(self.instruction, self.paragraph)
            self.finished.emit(text)
        except Exception as e:
            self.failed.emit(str(e))


# ============================================================
# Main window
# ============================================================

class SceneEditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scene Paragraph Editor")
        self.resize(1400, 900)

        self.current_path: Optional[str] = None
        self.paragraphs: List[str] = []
        self.ranges: List[tuple[int, int]] = []
        self.selected_index = 0
        self.dirty = False

        self.undo_buffers = defaultdict(lambda: deque(maxlen=10))

        self.llm = self._load_llm()
        self._build_ui()
        self._build_menu()

    # -----------------------------
    # Config / LLM
    # -----------------------------

    def _load_llm(self) -> LLMClient:
        with open("config.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        cfg = LLMConfig(
            provider=raw["llm_provider"],
            prompt_template=raw["prompt_template"],
            ollama_base_url=raw.get("ollama", {}).get("base_url", ""),
            ollama_model=raw.get("ollama", {}).get("model", ""),
            openai_model=raw.get("openai", {}).get("model", "gpt-4.1"),
            openai_api_key_env=raw.get("openai", {}).get("api_key_env", "OPENAI_API_KEY")
        )
        return LLMClient(cfg)

    # -----------------------------
    # UI
    # -----------------------------

    def _build_ui(self):
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)

        self.left_view = QtWidgets.QTextEdit(readOnly=True)
        self.left_view.setFont(QtGui.QFont("DejaVu Sans Mono", 11))
        self.left_view.viewport().installEventFilter(self)

        self.right_edit = QtWidgets.QTextEdit()
        self.right_edit.setFont(QtGui.QFont("DejaVu Sans Mono", 12))
        self.right_edit.textChanged.connect(self._mark_dirty)

        self.instr_box = QtWidgets.QLineEdit()
        self.instr_box.setPlaceholderText("Instruction → Enter to rewrite")
        self.instr_box.returnPressed.connect(self._on_rewrite)

        self.status = QtWidgets.QLabel("")

        right = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(right)
        v.addWidget(self.right_edit)
        v.addWidget(self.instr_box)
        v.addWidget(self.status)

        splitter.addWidget(self.left_view)
        splitter.addWidget(right)

        QtGui.QShortcut(QtGui.QKeySequence.Undo, self).activated.connect(self.undo_paragraph)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        m.addAction("Open", self.open_file)
        m.addAction("Save", self.save_file)
        m.addSeparator()
        m.addAction("Exit", self.close)

    # -----------------------------
    # File operations
    # -----------------------------

    def open_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open", "", "*.md")
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            self.paragraphs = split_paragraphs(f.read())

        self.current_path = path
        self.selected_index = 0
        self.undo_buffers.clear()
        self._rebuild_left()
        self._load_selected()
        self.dirty = False
        self._update_title()

    def save_file(self):
        self._commit_right()
        with open(self.current_path, "w", encoding="utf-8") as f:
            f.write(join_paragraphs(self.paragraphs))
        self.dirty = False
        self._update_title()
        self.status.setText("Saved.")

    # -----------------------------
    # Paragraph selection
    # -----------------------------

    def eventFilter(self, obj, event):
        if obj is self.left_view.viewport() and event.type() == QtCore.QEvent.MouseButtonPress:
            pos = self.left_view.cursorForPosition(event.position().toPoint()).position()
            for i, (s, e) in enumerate(self.ranges):
                if s <= pos <= e:
                    self._select_paragraph(i)
                    break
        return False

    def _select_paragraph(self, idx):
        if idx == self.selected_index:
            return
        self._push_undo()
        self._commit_right()
        self.selected_index = idx
        self._load_selected()

    # -----------------------------
    # Undo
    # -----------------------------

    def _push_undo(self):
        idx = self.selected_index
        cur = self.paragraphs[idx]
        buf = self.undo_buffers[idx]
        if not buf or buf[-1] != cur:
            buf.append(cur)

    def undo_paragraph(self):
        buf = self.undo_buffers[self.selected_index]
        if not buf:
            self.status.setText("Nothing to undo.")
            return

        prev = buf.pop()
        self.paragraphs[self.selected_index] = prev
        self.right_edit.setPlainText(prev)
        self._rebuild_left()
        self.status.setText("Undo applied.")
        self.dirty = True
        self._update_title()

    # -----------------------------
    # Rewrite
    # -----------------------------

    def _on_rewrite(self):
        instr = self.instr_box.text().strip()
        if not instr:
            return

        self._push_undo()
        para = self.paragraphs[self.selected_index]

        self.thread = QtCore.QThread()
        self.worker = RewriteWorker(self.llm, instr, para)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._apply_rewrite)
        self.worker.failed.connect(lambda e: self.status.setText(e))
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

    def _apply_rewrite(self, text):
        self.right_edit.setPlainText(text)
        self.instr_box.clear()
        self.status.setText("Rewrite applied.")
        self.dirty = True
        self._update_title()

    # -----------------------------
    # Sync helpers
    # -----------------------------

    def _commit_right(self):
        text = self.right_edit.toPlainText().rstrip()
        self.paragraphs[self.selected_index] = text
        self._rebuild_left()

    def _load_selected(self):
        self.right_edit.setPlainText(self.paragraphs[self.selected_index])
        self._highlight()

    def _rebuild_left(self):
        self.left_view.setPlainText(join_paragraphs(self.paragraphs))
        self.ranges = paragraph_char_ranges(self.paragraphs)
        self._highlight()

    def _highlight(self):
        s, e = self.ranges[self.selected_index]
        c = self.left_view.textCursor()
        c.setPosition(s)
        c.setPosition(e, QtGui.QTextCursor.KeepAnchor)
        sel = QtWidgets.QTextEdit.ExtraSelection()
        sel.cursor = c
        sel.format.setBackground(QtGui.QColor(255, 255, 0, 80))
        self.left_view.setExtraSelections([sel])

    def _mark_dirty(self):
        self.dirty = True
        self._update_title()

    def _update_title(self):
        name = os.path.basename(self.current_path) if self.current_path else ""
        self.setWindowTitle(f"Scene Paragraph Editor — {name}{' *' if self.dirty else ''}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    SceneEditorWindow().show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
