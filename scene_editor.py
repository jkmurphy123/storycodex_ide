import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional
from collections import deque, defaultdict
from functools import partial

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

    def __init__(self, llm: LLMClient, instruction: str, target_text: str):
        super().__init__()
        self.llm = llm
        self.instruction = instruction
        self.target_text = target_text

    @QtCore.Slot()
    def run(self):
        try:
            text = self.llm.rewrite(self.instruction, self.target_text)
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
        self.last_dir: Optional[str] = None
        self.recent_files: List[str] = []
        self.paragraphs: List[str] = []
        self.ranges: List[tuple[int, int]] = []
        self.selected_index = 0
        self.dirty = False

        self.undo_buffers = defaultdict(lambda: deque(maxlen=10))

        self.llm = self._load_llm()
        self.rewrite_in_flight = False
        self.rewrite_mode = None
        self.rewrite_selection = None
        self.style_buttons = []
        self._load_settings()
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

        self.style_scroll = QtWidgets.QScrollArea()
        self.style_scroll.setWidgetResizable(True)
        self.style_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.style_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.style_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.style_bar = QtWidgets.QWidget()
        self.style_layout = QtWidgets.QHBoxLayout(self.style_bar)
        self.style_layout.setContentsMargins(0, 0, 0, 0)
        self.style_layout.setSpacing(6)
        self.style_layout.addStretch(1)
        self.style_scroll.setWidget(self.style_bar)

        right = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(right)
        v.addWidget(self.style_scroll)
        v.addWidget(self.right_edit)
        v.addWidget(self.instr_box)
        v.addWidget(self.status)

        splitter.addWidget(self.left_view)
        splitter.addWidget(right)

        QtGui.QShortcut(QtGui.QKeySequence.Undo, self).activated.connect(self.undo_paragraph)
        self._build_style_buttons(self._load_styles())

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        m.addAction("Open", self.open_file)
        m.addAction("Save", self.save_file)
        self.recent_menu = m.addMenu("Recent...")
        self._update_recent_menu()
        m.addSeparator()
        m.addAction("Exit", self.close)

    def closeEvent(self, event):
        if not self.dirty:
            self._save_settings()
            event.accept()
            return

        resp = QtWidgets.QMessageBox.question(
            self,
            "Unsaved Changes",
            "You have unsaved changes. Save before quitting?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes,
        )

        if resp == QtWidgets.QMessageBox.Cancel:
            event.ignore()
            return

        if resp == QtWidgets.QMessageBox.Yes:
            if not self.current_path:
                path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", "", "*.md")
                if not path:
                    event.ignore()
                    return
                self.current_path = path
            self.save_file()

        self._save_settings()
        event.accept()

    def ensure_on_screen(self):
        try:
            screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
            if not screen:
                return
            geom = screen.availableGeometry()
            frame = self.frameGeometry()
            if geom.contains(frame.center()):
                return
            x = geom.x() + max(0, (geom.width() - frame.width()) // 2)
            y = geom.y() + max(0, (geom.height() - frame.height()) // 2)
            self.move(x, y)
        except Exception:
            pass

    # -----------------------------
    # File operations
    # -----------------------------

    def open_file(self):
        start_dir = self.last_dir or ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open", start_dir, "*.md")
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            self.paragraphs = split_paragraphs(f.read())

        self.current_path = path
        self.last_dir = os.path.dirname(path)
        self._add_recent_file(path)
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
        if self.current_path:
            self.last_dir = os.path.dirname(self.current_path)
            self._add_recent_file(self.current_path)

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
        if self.rewrite_in_flight:
            self.status.setText("Rewrite in progress.")
            return
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

    def _set_rewrite_controls_enabled(self, enabled: bool):
        self.right_edit.setEnabled(enabled)
        self.instr_box.setEnabled(enabled)
        for btn in self.style_buttons:
            btn.setEnabled(enabled)

    def _load_settings(self):
        settings = QtCore.QSettings("StoryCodex", "SceneParagraphEditor")
        self.last_dir = settings.value("last_dir", None)
        recent = settings.value("recent_files", [])
        if isinstance(recent, str):
            recent = [recent]
        self.recent_files = [p for p in recent if isinstance(p, str)]

    def _save_settings(self):
        settings = QtCore.QSettings("StoryCodex", "SceneParagraphEditor")
        settings.setValue("last_dir", self.last_dir)
        settings.setValue("recent_files", self.recent_files[:8])

    def _add_recent_file(self, path: str):
        if not path:
            return
        self.recent_files = [p for p in self.recent_files if p != path]
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:8]
        self._update_recent_menu()

    def _update_recent_menu(self):
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        if not self.recent_files:
            action = self.recent_menu.addAction("(No recent files)")
            action.setEnabled(False)
            return
        for path in self.recent_files:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda _=False, p=path: self._open_recent(p))

    def _open_recent(self, path: str):
        if not path or not os.path.exists(path):
            self.status.setText("Recent file not found.")
            self.recent_files = [p for p in self.recent_files if p != path]
            self._update_recent_menu()
            return
        if self.dirty:
            resp = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Save before opening another file?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if resp == QtWidgets.QMessageBox.Cancel:
                return
            if resp == QtWidgets.QMessageBox.Yes:
                if not self.current_path:
                    save_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save", "", "*.md")
                    if not save_path:
                        return
                    self.current_path = save_path
                self.save_file()
        with open(path, "r", encoding="utf-8") as f:
            self.paragraphs = split_paragraphs(f.read())
        self.current_path = path
        self.last_dir = os.path.dirname(path)
        self._add_recent_file(path)
        self.selected_index = 0
        self.undo_buffers.clear()
        self._rebuild_left()
        self._load_selected()
        self.dirty = False
        self._update_title()

    def _load_styles(self):
        styles_dir = os.path.join(os.getcwd(), "styles")
        if not os.path.isdir(styles_dir):
            return []

        styles = []
        for name in sorted(os.listdir(styles_dir)):
            if not name.endswith(".json"):
                continue
            path = os.path.join(styles_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception:
                continue
            label = raw.get("label")
            instruction = raw.get("instruction")
            if not label or not instruction:
                continue
            styles.append({
                "label": label,
                "instruction": instruction,
                "tooltip": raw.get("tooltip", "")
            })
        return styles

    def _build_style_buttons(self, styles):
        for btn in self.style_buttons:
            btn.deleteLater()
        self.style_buttons = []
        for style in styles:
            btn = QtWidgets.QPushButton(style["label"])
            if style.get("tooltip"):
                btn.setToolTip(style["tooltip"])
            btn.clicked.connect(partial(self._start_rewrite, style["instruction"]))
            self.style_layout.insertWidget(self.style_layout.count() - 1, btn)
            self.style_buttons.append(btn)

    def _get_rewrite_target(self):
        cursor = self.right_edit.textCursor()
        if cursor.hasSelection():
            selected = cursor.selectedText().replace("\u2029", "\n")
            if selected.strip():
                return "selection", selected, cursor.selectionStart(), cursor.selectionEnd()
        return "paragraph", self.paragraphs[self.selected_index], None, None

    def _start_rewrite(self, instruction: str):
        if self.rewrite_in_flight:
            self.status.setText("Rewrite in progress.")
            return

        instr = instruction.strip()
        if not instr:
            return

        mode, target_text, start, end = self._get_rewrite_target()
        self._push_undo()
        self.rewrite_mode = mode
        self.rewrite_selection = (start, end) if mode == "selection" else None
        self.rewrite_in_flight = True
        self._set_rewrite_controls_enabled(False)

        self.thread = QtCore.QThread()
        self.worker = RewriteWorker(self.llm, instr, target_text)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._apply_rewrite)
        self.worker.failed.connect(self._on_rewrite_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _on_rewrite(self):
        self._start_rewrite(self.instr_box.text())

    def _apply_rewrite(self, text):
        rewritten = text.strip()
        if not rewritten:
            self.status.setText("Empty rewrite output.")
            self._set_rewrite_controls_enabled(True)
            self.rewrite_in_flight = False
            self.rewrite_mode = None
            self.rewrite_selection = None
            return

        if self.rewrite_mode == "selection" and self.rewrite_selection:
            start, end = self.rewrite_selection
            cursor = self.right_edit.textCursor()
            cursor.beginEditBlock()
            cursor.setPosition(start)
            cursor.setPosition(end, QtGui.QTextCursor.KeepAnchor)
            cursor.insertText(rewritten)
            cursor.clearSelection()
            cursor.endEditBlock()
            self.right_edit.setTextCursor(cursor)
        else:
            self.right_edit.setPlainText(rewritten)

        self.instr_box.clear()
        self.status.setText("Rewrite applied.")
        self.dirty = True
        self._update_title()
        self._set_rewrite_controls_enabled(True)
        self.rewrite_in_flight = False
        self.rewrite_mode = None
        self.rewrite_selection = None

    def _on_rewrite_failed(self, error: str):
        self.status.setText(error)
        self._set_rewrite_controls_enabled(True)
        self.rewrite_in_flight = False
        self.rewrite_mode = None
        self.rewrite_selection = None

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
    window = SceneEditorWindow()
    window.show()
    QtCore.QTimer.singleShot(0, window.ensure_on_screen)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
