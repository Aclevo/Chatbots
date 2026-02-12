import sys
import os
import gi
import configparser
import threading
import re
import openvino_genai as ov
from datetime import datetime

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, Gio, GLib, Gdk, Pango

class ChatApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.Chatbots",
                         flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.config_path = "chatbots.conf"
        self.config = configparser.ConfigParser()
        self.load_settings()

    def load_settings(self):
        defaults = {
            'OpenSidebarWhenLaunched': 'y',
            'Model': 'Gemma 3',
            'Parameters': '4B',
            'Quantization': '4-bit',
            'Device': 'GPU'
        }
        if not os.path.exists(self.config_path):
            self.config['Settings'] = defaults
            with open(self.config_path, 'w') as f:
                self.config.write(f)
        else:
            self.config.read(self.config_path)
            if not self.config.has_section('Settings'):
                self.config.add_section('Settings')
            for key, val in defaults.items():
                if not self.config.has_option('Settings', key):
                    self.config.set('Settings', key, val)

    def save_settings(self, key, value):
        self.config.set('Settings', key, value)
        with open(self.config_path, 'w') as f:
            self.config.write(f)

    def do_activate(self):
        win = ChatWindow(application=self)
        win.present()

class ChatWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = kwargs['application']
        self.set_default_size(950, 650)
        self.pipeline = None
        self.current_model_path = ""
        self.current_chat_file = ""
        self.current_chat_title = ""
        self.selection_mode = False
        
        self.main_stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.set_content(self.main_stack)
        
        self.apply_styling()
        self.init_chat_page()
        self.init_settings_page()
        self.load_saved_chats()

    def apply_styling(self):
        css = Gtk.CssProvider()
        css.load_from_data("""
            .sidebar-pane { border-right: 1px solid alpha(currentColor, 0.1); }
            .chat-bubble { border-radius: 12px; margin-bottom: 6px; }
            .chat-bubble textview { background-color: transparent; }
            .user-bubble { background-color: @accent_bg_color; color: @accent_fg_color; }
            .bot-bubble { background-color: alpha(currentColor, 0.05); }
            .submenu-item { margin-left: 20px; }
            .selection-check { margin-right: 10px; }
            .selection-check > check { border-radius: 50%; }
            .destructive-action { color: @error_color; }
            .sidebar-header-box { padding: 8px; }
            .sidebar-title { font-weight: bold; }
        """.encode())
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    def init_chat_page(self):
        self.chat_view = Adw.OverlaySplitView()
        self.main_stack.add_named(self.chat_view, "chat")
        self.side_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, css_classes=["sidebar-pane"])
        self.chat_view.set_sidebar(self.side_box)
        
        self.sidebar_toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, css_classes=["sidebar-header-box"])
        self.side_box.append(self.sidebar_toolbar)
        self.sidebar_stack = Gtk.Stack()
        self.sidebar_stack.set_hexpand(True)
        self.sidebar_toolbar.append(self.sidebar_stack)
        
        nb = Gtk.Box(spacing=6)
        new_btn = Gtk.Button(icon_name="list-add-symbolic")
        new_btn.connect("clicked", self.on_new_chat)
        self.sel_btn = Gtk.Button(icon_name="object-select-symbolic")
        self.sel_btn.connect("clicked", self.toggle_selection_mode)
        set_btn = Gtk.Button(icon_name="emblem-system-symbolic")
        set_btn.connect("clicked", lambda _: self.main_stack.set_visible_child_name("settings"))
        
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        
        nb.append(new_btn); nb.append(self.sel_btn)
        nb.append(spacer); nb.append(set_btn)
        
        sb = Gtk.Box(spacing=6)
        trash_btn = Gtk.Button(icon_name="user-trash-symbolic", css_classes=["destructive-action"])
        trash_btn.connect("clicked", self.delete_selected_chats)
        ex_btn = Gtk.Button(icon_name="object-select-symbolic")
        ex_btn.connect("clicked", self.toggle_selection_mode)
        
        spacer2 = Gtk.Box()
        spacer2.set_hexpand(True)
        
        sb.append(trash_btn); sb.append(ex_btn)
        sb.append(spacer2)

        self.sidebar_stack.add_named(nb, "normal")
        self.sidebar_stack.add_named(sb, "select")
        
        self.chat_listbox = Gtk.ListBox(css_classes=["navigation-sidebar"])
        self.chat_listbox.set_vexpand(True)
        self.chat_listbox.connect("row-activated", self.on_chat_row_activated)
        scroll = Gtk.ScrolledWindow(child=self.chat_listbox)
        scroll.set_vexpand(True)
        self.side_box.append(scroll)

        ct = Adw.ToolbarView()
        self.chat_view.set_content(ct)
        mh = Adw.HeaderBar(title_widget=Adw.WindowTitle(title="Chatbots"), css_classes=["flat"])
        tb = Gtk.Button(icon_name="sidebar-show-symbolic")
        tb.connect("clicked", lambda _: self.chat_view.set_show_sidebar(not self.chat_view.get_show_sidebar()))
        mh.pack_start(tb); ct.add_top_bar(mh)

        cl = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        ct.set_content(cl)
        self.scroll = Gtk.ScrolledWindow()
        self.scroll.set_vexpand(True)
        self.chat_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.chat_container.set_margin_start(18)
        self.chat_container.set_margin_end(18)
        self.chat_container.set_margin_top(18)
        self.scroll.set_child(self.chat_container)
        cl.append(self.scroll)

        ib = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        ib.set_margin_start(12)
        ib.set_margin_end(12)
        ib.set_margin_bottom(12)
        self.entry = Gtk.Entry(placeholder_text="Type a message...")
        self.entry.set_hexpand(True)
        self.entry.connect("activate", self.on_send_clicked)
        ib.append(self.entry)
        s_btn = Gtk.Button(icon_name="go-up-symbolic", css_classes=["suggested-action"])
        s_btn.connect("clicked", self.on_send_clicked)
        ib.append(s_btn); cl.append(ib)

    def toggle_selection_mode(self, *args):
        self.selection_mode = not self.selection_mode
        self.sidebar_stack.set_visible_child_name("select" if self.selection_mode else "normal")
        row = self.chat_listbox.get_first_child()
        while row:
            box = row.get_child()
            check_box = box.get_first_child()
            label = check_box.get_next_sibling()
            check_box.set_visible(self.selection_mode)
            label.set_visible(True)
            row = row.get_next_sibling()

    def on_new_chat(self, *args):
        self.current_chat_file = ""
        self.current_chat_title = ""
        while self.chat_container.get_first_child():
            self.chat_container.remove(self.chat_container.get_first_child())

    def on_send_clicked(self, *args):
        text = self.entry.get_text().strip()
        if not text: return
        self.entry.set_text("")
        
        if not self.current_chat_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_chat_title = text[:50]
            self.current_chat_file = f"chats/{timestamp}.txt"
            os.makedirs("chats", exist_ok=True)
            
            with open(self.current_chat_file, "w") as f:
                f.write(f"TITLE: {self.current_chat_title}\n")
            
            self.add_chat_to_sidebar(self.current_chat_file, self.current_chat_title)
        
        self.add_bubble(text, True)
        with open(self.current_chat_file, "a") as f: 
            f.write(f"U: {text}\n")
        
        bl = self.add_bubble("...", False)
        threading.Thread(target=self.generate_response, args=(text, bl), daemon=True).start()

    def generate_response(self, prompt, view):
        try:
            conf = self.app.config['Settings']
            mdir = os.path.join(os.getcwd(), "models", f"{conf['Model'].replace(' ', '_')}-{conf['Parameters']}-{conf['Quantization']}")
            if self.pipeline is None or self.current_model_path != mdir:
                self.pipeline = ov.VLMPipeline(mdir, conf['Device'].upper())
                self.current_model_path = mdir
            gc = ov.GenerationConfig()
            gc.max_new_tokens = 1024
            fr = []
            def streamer(sub):
                fr.append(sub)
                GLib.idle_add(lambda: self.update_markdown_view(view, "".join(fr)))
                return False
            self.pipeline.generate(prompt, generation_config=gc, streamer=streamer)
            with open(self.current_chat_file, "a") as f: 
                f.write(f"B: {''.join(fr)}\n")
        except Exception as e:
            GLib.idle_add(lambda: self.update_markdown_view(view, f"Error: {str(e)}"))

    def update_markdown_view(self, view, text):
        buf = view.get_buffer()
        buf.set_text("")
        
        lines = text.split('\n')
        in_code = False
        code_lang = ""
        
        for line in lines:
            if line.startswith('```'):
                in_code = not in_code
                if in_code and len(line) > 3:
                    code_lang = line[3:].strip()
                else:
                    code_lang = ""
                continue
            
            tag = None
            clean_line = line
            
            if in_code:
                tag = "code"
            elif line.startswith('# '):
                tag = "h1"
                clean_line = line[2:]
            elif line.startswith('## '):
                tag = "h2"
                clean_line = line[3:]
            elif line.startswith('### '):
                tag = "h3"
                clean_line = line[4:]
            elif line.startswith('#### '):
                tag = "h4"
                clean_line = line[5:]
            elif line.startswith('- ') or line.startswith('* '):
                tag = "bullet"
                clean_line = "• " + line[2:]
            elif re.match(r'^\d+\.\s', line):
                tag = "numbered"
            elif line.startswith('> '):
                tag = "quote"
                clean_line = line[2:]
            
            start_iter = buf.get_end_iter()
            buf.insert(start_iter, clean_line + "\n")
            end_iter = buf.get_end_iter()
            
            if tag:
                buf.apply_tag_by_name(tag, start_iter, end_iter)
            
            self.apply_inline_markdown(buf, start_iter, end_iter)
        
        adj = self.scroll.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    def apply_inline_markdown(self, buf, start, end):
        text = buf.get_text(start, end, False)
        
        offset = start.get_offset()
        
        for match in re.finditer(r'\*\*\*(.+?)\*\*\*', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("bold_italic", s, e)
        
        for match in re.finditer(r'\*\*(.+?)\*\*', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("bold", s, e)
        
        for match in re.finditer(r'\*(.+?)\*', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("italic", s, e)
        
        for match in re.finditer(r'_(.+?)_', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("italic", s, e)
        
        for match in re.finditer(r'`(.+?)`', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("inline_code", s, e)
        
        for match in re.finditer(r'~~(.+?)~~', text):
            s = buf.get_iter_at_offset(offset + match.start())
            e = buf.get_iter_at_offset(offset + match.end())
            buf.apply_tag_by_name("strikethrough", s, e)

    def add_bubble(self, text, is_user):
        view = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        view.set_top_margin(8)
        view.set_bottom_margin(8)
        view.set_left_margin(12)
        view.set_right_margin(12)
        view.add_css_class("chat-bubble")
        view.add_css_class("user-bubble" if is_user else "bot-bubble")
        
        buf = view.get_buffer()
        buf.create_tag("h1", weight=Pango.Weight.BOLD, scale=1.5)
        buf.create_tag("h2", weight=Pango.Weight.BOLD, scale=1.3)
        buf.create_tag("h3", weight=Pango.Weight.BOLD, scale=1.15)
        buf.create_tag("h4", weight=Pango.Weight.BOLD, scale=1.1)
        buf.create_tag("bold", weight=Pango.Weight.BOLD)
        buf.create_tag("italic", style=Pango.Style.ITALIC)
        buf.create_tag("bold_italic", weight=Pango.Weight.BOLD, style=Pango.Style.ITALIC)
        buf.create_tag("code", family="monospace", background="alpha(currentColor, 0.1)")
        buf.create_tag("inline_code", family="monospace", background="alpha(currentColor, 0.1)")
        buf.create_tag("bullet", left_margin=20)
        buf.create_tag("numbered", left_margin=20)
        buf.create_tag("quote", left_margin=20, foreground="alpha(currentColor, 0.7)", style=Pango.Style.ITALIC)
        buf.create_tag("strikethrough", strikethrough=True)
        
        if is_user:
            buf.set_text(text)
        else:
            self.update_markdown_view(view, text)
        
        wrapper = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        
        if is_user:
            left_spacer = Gtk.Box()
            left_spacer.set_hexpand(True)
            wrapper.append(left_spacer)
            wrapper.append(view)
        else:
            wrapper.append(view)
            right_spacer = Gtk.Box()
            right_spacer.set_hexpand(True)
            wrapper.append(right_spacer)
        
        self.chat_container.append(wrapper)
        return view

    def add_chat_to_sidebar(self, fp, title):
        rc = Gtk.Box(spacing=10)
        ck = Gtk.CheckButton(visible=self.selection_mode, css_classes=["selection-check"])
        lb = Gtk.Label(label=title, xalign=0)
        lb.set_hexpand(True)
        lb.set_visible(True)
        rc.append(ck)
        rc.append(lb)
        r = Gtk.ListBoxRow(child=rc)
        r.filepath = fp
        r.chat_title = title
        self.chat_listbox.prepend(r)

    def load_saved_chats(self):
        if os.path.exists("chats"):
            for f in sorted([f for f in os.listdir("chats") if f.endswith(".txt")], reverse=True):
                filepath = f"chats/{f}"
                title = "Untitled Chat"
                
                try:
                    with open(filepath, "r") as file:
                        first_line = file.readline().strip()
                        if first_line.startswith("TITLE: "):
                            title = first_line[7:]
                        else:
                            for line in file:
                                if line.startswith("U: "):
                                    title = line[3:].strip()[:50]
                                    break
                except:
                    pass
                
                self.add_chat_to_sidebar(filepath, title)

    def on_chat_row_activated(self, box, r):
        if self.selection_mode: 
            check = r.get_child().get_first_child()
            check.set_active(not check.get_active())
            return
        
        while self.chat_container.get_first_child():
            self.chat_container.remove(self.chat_container.get_first_child())
        
        self.current_chat_file = r.filepath
        self.current_chat_title = r.chat_title
        
        if os.path.exists(r.filepath):
            with open(r.filepath, "r") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln.startswith("TITLE: "):
                        continue
                    elif ln.startswith("U: "):
                        self.add_bubble(ln[3:], True)
                    elif ln.startswith("B: "):
                        self.add_bubble(ln[3:], False)

    def delete_selected_chats(self, *args):
        r = self.chat_listbox.get_first_child()
        td, acd = [], False
        while r:
            if r.get_child().get_first_child().get_active():
                td.append(r)
                if r.filepath == self.current_chat_file: acd = True
            r = r.get_next_sibling()
        for r in td:
            if os.path.exists(r.filepath): os.remove(r.filepath)
            self.chat_listbox.remove(r)
        if acd or not self.chat_listbox.get_first_child(): self.on_new_chat()
        self.toggle_selection_mode()

    def init_settings_page(self):
        self.settings_view = Adw.NavigationSplitView()
        self.main_stack.add_named(self.settings_view, "settings")
        self.settings_view.set_sidebar(Adw.NavigationPage(child=self.build_settings_sidebar(), title="Settings"))
        
        st = Adw.ToolbarView()
        sh = Adw.HeaderBar(title_widget=Adw.WindowTitle(title="Chatbots"))
        st.add_top_bar(sh)
        self.pref_stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        st.set_content(self.pref_stack)
        self.settings_view.set_content(Adw.NavigationPage(child=st, title="Chatbots"))
        
        gp = Adw.PreferencesPage(); gg = Adw.PreferencesGroup()
        sw = Gtk.Switch(active=(self.app.config.get('Settings', 'OpenSidebarWhenLaunched') == 'y'), valign=Gtk.Align.CENTER)
        sw.connect("state-set", lambda _, s: self.app.save_settings('OpenSidebarWhenLaunched', 'y' if s else 'n'))
        rw = Adw.ActionRow(title="Open sidebar when launched")
        rw.add_suffix(sw); gg.add(rw); gp.add(gg); self.pref_stack.add_named(gp, "General")
        
        tp = Adw.PreferencesPage(); tg = Adw.PreferencesGroup()
        sm = [("Model", ["Gemma 3"]), ("Parameters", ["4B"]), ("Quantization", ["4-bit"]), ("Device", ["GPU", "CPU"])]
        for k, o in sm:
            r = Adw.ComboRow(title=k, model=Gtk.StringList.new(o))
            r.set_selected(o.index(self.app.config['Settings'].get(k, o[0])))
            r.connect("notify::selected", lambda r, _, key=k, opts=o: self.app.save_settings(key, opts[r.get_selected()]))
            tg.add(r)
        tp.add(tg); self.pref_stack.add_named(tp, "Model")

    def build_settings_sidebar(self):
        b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, css_classes=["sidebar-pane"])
        hb = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, css_classes=["sidebar-header-box"], spacing=6)
        back = Gtk.Button(icon_name="go-previous-symbolic")
        back.connect("clicked", lambda _: self.main_stack.set_visible_child_name("chat"))
        title = Gtk.Label(label="Settings", css_classes=["sidebar-title"])
        title.set_hexpand(True)
        title.set_xalign(0.5)
        hb.append(back); hb.append(title); b.append(hb)
        
        lb = Gtk.ListBox(css_classes=["navigation-sidebar"])
        lb.set_vexpand(True)
        for n in ["General", "Model"]: 
            lb.append(Adw.ActionRow(title=n, activatable=True))
        lb.connect("row-activated", lambda l, r: self.pref_stack.set_visible_child_name(r.get_title()))
        b.append(lb); return b

if __name__ == "__main__":
    app = ChatApp()
    app.run(sys.argv)
