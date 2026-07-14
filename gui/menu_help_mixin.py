"""
Menu bar, Help window, About dialog and info popups.

@author: Sharad Kumar Gupta
"""
from gui._deps import *  # noqa: F401,F403 - reproduce original module namespace


# ── Software metadata (keep in sync with README.md) ────────────────────────────
PARACUDA_AUTHOR = "Sharad Kumar Gupta"
PARACUDA_REPO_URL = "https://github.com/sharadgupta27/paracuda"
PARACUDA_CONTACT = "sharadgupta27@gmail.com"
PARACUDA_LICENSE = "Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)"

# ── What's New — changes since the last release ────────────────────────────────
# Newest release first. Each entry: (heading, [bullet, ...]).
WHATS_NEW = [
    ("Performance", [
        "Typing in the Min/Max Wave fields no longer stutters - the live "
        "process-flow preview used to rebuild itself on every keystroke, which "
        "could stall the window and drop characters.",
        "Batch runs no longer redo Target Variable Outlier Removal for every "
        "model - it is now computed once per property and reused, instead of "
        "being recomputed (and re-logged) for each model in the batch.",
    ]),
    ("Preprocessing", [
        "New Target Variable Outlier Removal: drops samples whose selected "
        "property value (e.g. clay) is a z-score or IQR outlier. Enable it with "
        "the checkbox in Step 3 - Preprocessing.",
        "Find Best Preprocessing now works with several properties selected at "
        "once. Each property is searched independently and gets its own winning "
        "preprocessing + model, with a summary table at the end.",
    ]),
    ("Plots & Visualization", [
        "Training-data scatter plots are now produced alongside the test "
        "scatter, both in the Visualization tab and in the exported PDF. Titles "
        "name the split ('Training Data' / 'Test Data').",
        "The Reflectance Spectra viewer lets you choose how many spectra to "
        "show - drawn at random, or from a specific sample range.",
        "Spectra are drawn with discrete colours instead of a gradient, so "
        "individual curves stay distinguishable.",
        "The per-sample legend is omitted above 20 spectra, where it covered "
        "the curves.",
        "All plots now draw the full box (top and right axes).",
    ]),
    ("Usability", [
        "Starting a Run or Batch Run automatically switches to the Status & Log "
        "tab, so progress is visible as it happens.",
        "The log now explains why Save Model is disabled for multi-property "
        "runs instead of leaving the button greyed out without a reason.",
        "Help -> About Paracuda III now shows the creator, a link to the source "
        "code, and contact information.",
    ]),
]


class MenuHelpMixin:
    
    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(
            label="🔄 Data Converter",
            command=self._open_data_converter,
        )

        # View menu — colour theme selection (applies to the whole software).
        try:
            from paracuda_theme import list_palettes
            palettes = list_palettes()
        except Exception:
            palettes = []
        if palettes:
            view_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="View", menu=view_menu)
            theme_menu = tk.Menu(view_menu, tearoff=0)
            view_menu.add_cascade(label="🎨 Theme", menu=theme_menu)
            self._theme_menu_var = tk.StringVar(
                value=getattr(self, 'theme_name', palettes[0]))
            for name in palettes:
                theme_menu.add_radiobutton(
                    label=name, value=name, variable=self._theme_menu_var,
                    command=lambda n=name: self._set_theme(n))

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="🤖 Ask Help Assistant",
                              command=self.show_help)
        help_menu.add_separator()

        # Entry-point topics worth a direct shortcut. Everything else (tabular
        # prediction, resampling, image processing, harmonization, baseline
        # correction, missing data, hyperparameter tuning, the flow chart,
        # model portability, overfitting, ...) is a keyword away inside
        # "Ask Help Assistant" itself, so it does not need its own menu entry.
        help_menu.add_command(label="📖 Getting Started",
                              command=lambda: self.show_help_topic("getting_started"))
        help_menu.add_command(label="📂 Loading Data",
                              command=lambda: self.show_help_topic("data_loading"))
        help_menu.add_command(label="🗺️ Complete Workflow Guide",
                              command=lambda: self.show_help_topic("workflow"))
        help_menu.add_command(label="✅ Best Practices",
                              command=lambda: self.show_help_topic("best_practices"))
        help_menu.add_command(label="🛠 Troubleshooting",
                              command=lambda: self.show_help_topic("troubleshooting"))
        help_menu.add_separator()
        help_menu.add_command(label="🆕 What's New",
                              command=self.show_whats_new)
        help_menu.add_command(label="ℹ️ About Paracuda III",
                              command=self.show_about)

    def _set_theme(self, name):
        """Switch the active colour theme live and persist the choice.

        Re-applies the ttk styles, repaints the plain-``tk`` widgets whose
        backgrounds are set directly (canvases, the log console, the root
        window), redraws the process-flow preview, and saves the choice so the
        Data Converter (a separate process) picks it up on its next launch.
        """
        try:
            from paracuda_theme import (get_palette, save_theme_name,
                                        apply_ttk_theme)
        except Exception:
            return
        self.theme_name = name
        self._palette = apply_ttk_theme(self, get_palette(name))
        save_theme_name(name)

        # Repaint tk (non-ttk) widgets that carry an explicit background.
        try:
            self.configure(bg=self._c('WIN'))
        except Exception:
            pass
        for attr in ('params_canvas',):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.configure(bg=self._c('WIN'))
                except Exception:
                    pass
        # Every wizard-step scroll canvas.
        try:
            for canv in self._scroll_canvases:
                canv.configure(bg=self._c('WIN'))
        except Exception:
            pass
        if hasattr(self, 'status_text'):
            try:
                self.status_text.configure(bg=self._c('BNR'))
            except Exception:
                pass
        # Redraw the process-flow preview with the new palette colours.
        if hasattr(self, 'refresh_flow_preview'):
            try:
                self.refresh_flow_preview()
            except Exception:
                pass
        # Let the user know the converter updates on next open.
        if hasattr(self, 'status_text'):
            self.status_text.insert(
                tk.END, f"Theme changed to '{name}'. "
                        "The Data Converter will use it next time it is opened.\n")
            self.status_text.see(tk.END)

    def show_help_topic(self, topic_id: str):
        """Open the help assistant pre-loaded with a specific topic."""
        try:
            self.show_help(prefill_topic=topic_id)
        except Exception:
            self.show_help()

    def _add_link_label(self, parent, text, url, **grid_kw):
        """A clickable, link-styled label that opens ``url`` in the browser."""
        import webbrowser

        link = ttk.Label(parent, text=text, foreground="#1a6fb5",
                         font=('Helvetica', 10, 'underline'), cursor="hand2")
        link.grid(**grid_kw)
        link.bind("<Button-1>", lambda _e: webbrowser.open_new(url))
        return link

    def show_about(self):
        """Show About dialog: what the software is, who made it, and how to reach them."""
        win = tk.Toplevel(self)
        win.title("About Paracuda III")
        win.resizable(False, False)
        win.transient(self)
        frm = ttk.Frame(win, padding=24)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Paracuda III", font=('Helvetica', 18, 'bold')).pack()
        ttk.Label(frm, text="Spectral Analysis Tool",
                  font=('Helvetica', 11)).pack(pady=(2, 12))

        about_text = (
            "Paracuda III is a hyperspectral data analysis platform\n"
            "for soil, vegetation, and other spectroscopy data.\n\n"
            "Features:\n"
            "  • Multi-model regression (PLS, RF, XGBoost, SVR, Ridge, …)\n"
            "  • Cross-validation with overfitting detection\n"
            "  • Batch processing across properties and models\n"
            "  • Spectral preprocessing (10+ methods)\n"
            "  • Hyperspectral image prediction (GeoTIFF)\n"
            "  • Tabular prediction for unknown samples\n"
            "  • Spectral harmonization / transfer functions\n"
            "  • Data integrity & randomization tests\n"
            "  • Portable model files with preprocessing\n\n"
            "Developed for applied spectroscopy research."
        )
        ttk.Label(frm, text=about_text, justify="left").pack(anchor="w")

        ttk.Separator(frm, orient='horizontal').pack(fill='x', pady=14)

        # ── Creator, source code, contact ──────────────────────────────────
        info = ttk.Frame(frm)
        info.pack(anchor="w", fill="x")

        ttk.Label(info, text="Created by:", font=('Helvetica', 10, 'bold')).grid(
            row=0, column=0, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(info, text=PARACUDA_AUTHOR, font=('Helvetica', 10)).grid(
            row=0, column=1, sticky="w", pady=2)

        ttk.Label(info, text="Source code:", font=('Helvetica', 10, 'bold')).grid(
            row=1, column=0, sticky="w", padx=(0, 12), pady=2)
        self._add_link_label(info, PARACUDA_REPO_URL, PARACUDA_REPO_URL,
                             row=1, column=1, sticky="w", pady=2)

        ttk.Label(info, text="Contact:", font=('Helvetica', 10, 'bold')).grid(
            row=2, column=0, sticky="w", padx=(0, 12), pady=2)
        self._add_link_label(info, PARACUDA_CONTACT, f"mailto:{PARACUDA_CONTACT}",
                             row=2, column=1, sticky="w", pady=2)

        ttk.Label(info, text="Licence:", font=('Helvetica', 10, 'bold')).grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=2)
        ttk.Label(info, text=PARACUDA_LICENSE, font=('Helvetica', 9),
                  wraplength=380, justify="left").grid(
            row=3, column=1, sticky="w", pady=2)

        ttk.Button(frm, text="Close", command=win.destroy).pack(pady=(16, 0))

    def show_whats_new(self):
        """Show the What's New dialog: changes since the last release, as bullets."""
        win = tk.Toplevel(self)
        win.title("What's New — Paracuda III")
        win.geometry("640x520")
        win.minsize(520, 380)
        win.transient(self)

        frm = ttk.Frame(win, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="What's New", font=('Helvetica', 16, 'bold')).pack(anchor="w")
        ttk.Label(frm, text="Changes since the last release",
                  font=('Helvetica', 10), foreground="#666666").pack(anchor="w", pady=(2, 10))

        body = ttk.Frame(frm)
        body.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(body, orient="vertical")
        scroll.pack(side="right", fill="y")

        txt = tk.Text(body, wrap="word", yscrollcommand=scroll.set,
                      font=('Segoe UI', 10), padx=10, pady=10,
                      relief="flat", borderwidth=1, highlightthickness=1,
                      highlightbackground="#d0d0d0")
        txt.pack(side="left", fill="both", expand=True)
        scroll.config(command=txt.yview)

        txt.tag_configure("heading", font=('Segoe UI', 11, 'bold'),
                          spacing1=10, spacing3=4)
        txt.tag_configure("bullet", lmargin1=14, lmargin2=30, spacing3=4)

        for heading, bullets in WHATS_NEW:
            txt.insert(tk.END, f"{heading}\n", "heading")
            for bullet in bullets:
                txt.insert(tk.END, f"•  {bullet}\n", "bullet")

        txt.config(state="disabled")  # read-only

        ttk.Button(frm, text="Close", command=win.destroy).pack(pady=(12, 0))
    
    def show_help(self, prefill_topic: str = None):
        """Show interactive help dialog with AI assistant"""
        try:
            # Create help dialog
            help_dialog = tk.Toplevel(self)
            help_dialog.title("Paracuda Help Assistant")
            help_dialog.geometry("920x680")
            help_dialog.minsize(720, 500)
            help_dialog.transient(self)
            help_dialog.configure(bg="#f5f7fa")

            # ── Fonts ──────────────────────────────────────────────────
            FONT_TITLE   = ('Segoe UI', 15, 'bold')
            FONT_HEADING = ('Segoe UI', 11, 'bold')
            FONT_BODY    = ('Segoe UI', 10)
            FONT_ENTRY   = ('Segoe UI', 10)
            FONT_BTN     = ('Segoe UI', 10)

            # ── Outer padding frame ────────────────────────────────────
            main_frame = ttk.Frame(help_dialog, padding="16 14 16 10")
            main_frame.grid(row=0, column=0, sticky="nsew")
            help_dialog.grid_columnconfigure(0, weight=1)
            help_dialog.grid_rowconfigure(0, weight=1)
            main_frame.grid_columnconfigure(0, weight=1)
            main_frame.grid_rowconfigure(2, weight=1)

            # ── Title bar ─────────────────────────────────────────────
            title_frame = ttk.Frame(main_frame)
            title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
            title_frame.grid_columnconfigure(0, weight=1)
            ttk.Label(title_frame, text="🤖  Paracuda Help Assistant",
                      font=FONT_TITLE, foreground="#1a3a5c").grid(row=0, column=0, sticky="w")
            ttk.Separator(main_frame, orient='horizontal').grid(
                row=0, column=0, sticky="ew", pady=(42, 0))

            # ── Query input row ────────────────────────────────────────
            query_frame = ttk.Frame(main_frame)
            query_frame.grid(row=1, column=0, sticky="ew", pady=(8, 10))
            query_frame.grid_columnconfigure(1, weight=1)

            ttk.Label(query_frame, text="Ask a question:", font=FONT_HEADING,
                      foreground="#1a3a5c").grid(row=0, column=0, sticky="w", padx=(0, 10))
            query_entry = ttk.Entry(query_frame, font=FONT_ENTRY)
            query_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))
            query_entry.focus()

            ask_btn = ttk.Button(query_frame, text="Ask", width=8)
            ask_btn.grid(row=0, column=2, padx=(0, 4))
            clear_btn = ttk.Button(query_frame, text="Clear", width=8)
            clear_btn.grid(row=0, column=3)

            # ── Response area ──────────────────────────────────────────
            response_frame = ttk.Frame(main_frame)
            response_frame.grid(row=2, column=0, sticky="nsew")
            response_frame.grid_columnconfigure(0, weight=1)
            response_frame.grid_rowconfigure(0, weight=1)

            response_text = tk.Text(
                response_frame,
                wrap=tk.WORD,
                font=FONT_BODY,
                relief=tk.FLAT,
                borderwidth=0,
                padx=12,
                pady=10,
                bg="#ffffff",
                fg="#222222",
                selectbackground="#c8d8f0",
                cursor="arrow",
                spacing1=2,   # pixels above each line
                spacing3=2,   # pixels below each line
            )
            response_text.grid(row=0, column=0, sticky="nsew")

            scrollbar = ttk.Scrollbar(response_frame, orient=tk.VERTICAL,
                                      command=response_text.yview)
            scrollbar.grid(row=0, column=1, sticky="ns")
            response_text.config(yscrollcommand=scrollbar.set)

            # Add a subtle border around the text area
            response_text.config(highlightbackground="#c8d0de", highlightthickness=1)

            # ── Text tags ─────────────────────────────────────────────
            response_text.tag_configure("title",
                font=('Segoe UI', 13, 'bold'), foreground="#1a3a5c",
                spacing1=8, spacing3=4)
            response_text.tag_configure("section",
                font=('Segoe UI', 11, 'bold'), foreground="#1a4a7a",
                spacing1=6, spacing3=2)
            response_text.tag_configure("bullet",
                font=FONT_BODY, lmargin1=22, lmargin2=38, spacing1=1)
            response_text.tag_configure("code",
                font=('Consolas', 9), background="#eef2f8",
                foreground="#333333", relief=tk.FLAT)
            response_text.tag_configure("info_button",
                foreground="#0066cc", font=('Segoe UI', 9))
            response_text.tag_configure("clickable",
                foreground="#0066cc", underline=True)
            response_text.tag_configure("body",
                font=FONT_BODY, spacing1=1)

            # ── Populate initial content ───────────────────────────────
            if prefill_topic:
                initial_help = self.help_assistant.find_best_match(prefill_topic)
            else:
                initial_help = self.help_assistant.find_best_match("")
            self._display_formatted_help(response_text, initial_help)
            response_text.config(state=tk.DISABLED)

            # ── Bottom button bar ──────────────────────────────────────
            sep2 = ttk.Separator(main_frame, orient='horizontal')
            sep2.grid(row=3, column=0, sticky="ew", pady=(10, 6))

            btn_frame = ttk.Frame(main_frame)
            btn_frame.grid(row=4, column=0, sticky="e")
            close_btn = ttk.Button(btn_frame, text="Close", width=10,
                                   command=help_dialog.destroy)
            close_btn.pack()

            # ── Logic ──────────────────────────────────────────────────
            def on_ask():
                query = query_entry.get().strip()
                response_text.config(state=tk.NORMAL)
                response_text.delete(1.0, tk.END)
                response = self.help_assistant.find_best_match(query)
                self._display_formatted_help(response_text, response)
                response_text.config(state=tk.DISABLED)
                response_text.see(1.0)

            ask_btn.config(command=on_ask)
            clear_btn.config(command=lambda: query_entry.delete(0, tk.END))
            query_entry.bind('<Return>', lambda e: on_ask())

            # ── Centre on screen ───────────────────────────────────────
            help_dialog.update_idletasks()
            w = help_dialog.winfo_width()
            h = help_dialog.winfo_height()
            x = (help_dialog.winfo_screenwidth() // 2) - (w // 2)
            y = (help_dialog.winfo_screenheight() // 2) - (h // 2)
            help_dialog.geometry(f'{w}x{h}+{x}+{y}')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open help: {str(e)}")
    
    def _display_formatted_help(self, text_widget, content):
        """Display help content with formatting and clickable info buttons"""
        lines = content.split('\n')
        
        # Keywords that should have info buttons
        model_keywords = ["PLS-R", "SVM", "Random Forest", "XGBoost", "Ridge", "Lasso", 
                         "Elastic Net", "Huber", "Gradient Boosting", "Gaussian Process", 
                         "Linear Regression", "Multiple Linear"]
        cv_keywords = ["K-Fold", "Leave-One-Out", "Leave-P-Out", "cross validation", "Cross-Validation"]
        metric_keywords = ["R²", "RMSE", "R-Squared", "Root Mean Square"]
        
        for line in lines:
            if line and line[0] != ' ' and '=' in line and '=' * 10 in line:
                # This is an underline for a title, skip it
                continue
            elif line and not line.startswith(' ') and ':' not in line[:50] and len(line) < 60:
                # This might be a title
                if line.strip() and not line.startswith('•') and not line.startswith('-'):
                    text_widget.insert(tk.END, line + '\n', "title")
                else:
                    text_widget.insert(tk.END, line + '\n')
            elif line.strip().startswith('**') and line.strip().endswith('**'):
                # Bold section headers - check for keywords and add info buttons
                clean_line = line.replace('**', '').strip()
                text_widget.insert(tk.END, clean_line, "section")
                
                # Check if this is a model/CV/metric name and add info button
                if any(keyword in clean_line for keyword in model_keywords):
                    text_widget.insert(tk.END, " ")
                    self._insert_info_button(text_widget, "models", "📘 More about models")
                elif any(keyword in clean_line for keyword in cv_keywords):
                    text_widget.insert(tk.END, " ")
                    self._insert_info_button(text_widget, "cross_validation", "📘 More about CV")
                elif any(keyword in clean_line for keyword in metric_keywords):
                    text_widget.insert(tk.END, " ")
                    self._insert_info_button(text_widget, "metrics", "📘 More about metrics")
                
                text_widget.insert(tk.END, '\n')
            elif line.strip().startswith('•') or line.strip().startswith('-'):
                # Bullet points
                text_widget.insert(tk.END, line + '\n', "bullet")
            elif line.strip().startswith('✓'):
                # Checkmark bullets
                text_widget.insert(tk.END, line + '\n', "bullet")
            else:
                text_widget.insert(tk.END, line + '\n')
    
    def _insert_info_button(self, text_widget, info_type, button_text):
        """Insert a clickable info button in the text widget"""
        tag_name = f"info_{info_type}_{id(text_widget)}"
        text_widget.insert(tk.END, button_text, tag_name)
        
        # Make it clickable
        text_widget.tag_config(tag_name, foreground="#0066cc", underline=True)
        text_widget.tag_bind(tag_name, "<Button-1>", lambda e: self.show_info(info_type))
        text_widget.tag_bind(tag_name, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind(tag_name, "<Leave>", lambda e: text_widget.config(cursor=""))
    
    def show_info(self, info_type):
        """Show quick information with URLs for models, CV methods, or metrics"""
        info_data = {
            "models": {
                "title": "Regression Models - Quick Reference",
                "content": """
**PLS-R (Partial Least Squares Regression)**
Best for high-dimensional spectral data with many correlated features.
📚 Learn more: https://scikit-learn.org/stable/modules/cross_decomposition.html

**SVM (Support Vector Machine)**
Non-linear regression with kernel trick. Good for complex patterns.
📚 Learn more: https://scikit-learn.org/stable/modules/svm.html#regression

**Random Forest**
Ensemble method with high accuracy and feature importance.
📚 Learn more: https://scikit-learn.org/stable/modules/ensemble.html#forest

**XGBoost**
Gradient boosting for highest accuracy. Industry standard.
📚 Learn more: https://xgboost.readthedocs.io/

**Ridge Regression**
Linear regression with L2 regularization to prevent overfitting.
📚 Learn more: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression

**Lasso Regression**
Linear regression with L1 regularization for feature selection.
📚 Learn more: https://scikit-learn.org/stable/modules/linear_model.html#lasso

**Elastic Net**
Combines L1 and L2 regularization. Balance between Ridge and Lasso.
📚 Learn more: https://scikit-learn.org/stable/modules/linear_model.html#elastic-net

**Huber Regressor**
Robust to outliers using a combination of squared and absolute loss.
📚 Learn more: https://scikit-learn.org/stable/modules/linear_model.html#huber-regression

**Gradient Boosting**
Sequential ensemble method building trees to correct previous errors.
📚 Learn more: https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting

**Gaussian Process**
Probabilistic model providing uncertainty estimates with predictions.
📚 Learn more: https://scikit-learn.org/stable/modules/gaussian_process.html

**Multiple Linear Regression**
Simple linear regression for baseline comparisons.
📚 Learn more: https://scikit-learn.org/stable/modules/linear_model.html

💡 For spectral data: Start with PLS-R, then compare with Random Forest or XGBoost.
"""
            },
            "cross_validation": {
                "title": "Cross-Validation Methods",
                "content": """
**K-Fold Cross-Validation**
Splits data into K folds, trains on K-1, tests on remaining fold.
• Recommended: K=5 or K=10
• Use when: Moderate to large datasets (>50 samples)
📚 Learn more: https://scikit-learn.org/stable/modules/cross_validation.html#k-fold

**Leave-One-Out (LOO)**
Leaves one sample out for testing, trains on all others.
• Use when: Small datasets (<50 samples)
• Warning: Computationally expensive for large datasets
📚 Learn more: https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out

**Leave-P-Out**
Leaves P samples out for testing. More thorough than LOO.
• Use when: Very small datasets requiring rigorous validation
• Warning: Very computationally intensive
📚 Learn more: https://scikit-learn.org/stable/modules/cross_validation.html#leave-p-out

**Metrics to Watch:**
• R² (CV): Should be close to training R²
• RMSE (CV): Should be similar to training RMSE
• Large difference indicates overfitting

📘 Complete Guide: https://scikit-learn.org/stable/modules/cross_validation.html
"""
            },
            "metrics": {
                "title": "Performance Metrics Explained",
                "content": """
**R² (R-Squared / Coefficient of Determination)**
Measures the proportion of variance explained by the model.
• Range: 0 to 1 (higher is better)
• 0.9-1.0: Excellent    • 0.7-0.9: Good
• 0.5-0.7: Moderate     • <0.5: Poor
📚 Learn more: https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

**RMSE (Root Mean Square Error)**
Average magnitude of prediction errors in original units.
• Lower is better
• Compare to property range (e.g., RMSE < 5% for Clay)
📚 Learn more: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error

**Training vs CV Scores:**
• Similar scores: Model generalizes well ✓
• Training >> CV: Overfitting (model memorizes data) ✗
• Solution: Simplify model or collect more data

📘 All Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
"""
            }
        }
        
        if info_type not in info_data:
            return
        
        info = info_data[info_type]
        
        # Create info dialog
        info_dialog = tk.Toplevel(self)
        info_dialog.title(info["title"])
        info_dialog.geometry("650x550")
        info_dialog.transient(self)
        
        # Main frame
        main_frame = ttk.Frame(info_dialog, padding="15")
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text=info["title"], 
                               font=('Arial', 12, 'bold'), foreground='#1a5490')
        title_label.pack(pady=(0, 10))
        
        # Text area with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill="both", expand=True)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Arial', 10),
                             relief=tk.SUNKEN, borderwidth=2)
        text_widget.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                 command=text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Configure tags
        text_widget.tag_configure("bold", font=('Arial', 10, 'bold'))
        text_widget.tag_configure("url", foreground="blue", underline=True)
        
        # Insert formatted content
        lines = info["content"].strip().split('\n')
        for line in lines:
            if line.strip().startswith('**') and line.strip().endswith('**'):
                clean_line = line.replace('**', '')
                text_widget.insert(tk.END, clean_line + '\n', "bold")
            elif 'http' in line:
                # Parse URLs
                parts = line.split('http')
                text_widget.insert(tk.END, parts[0])
                for i, part in enumerate(parts[1:], 1):
                    url_end = part.find(' ')
                    if url_end == -1:
                        url_end = len(part)
                    url = 'http' + part[:url_end]
                    text_widget.insert(tk.END, url, "url")
                    text_widget.insert(tk.END, part[url_end:])
                text_widget.insert(tk.END, '\n')
            else:
                text_widget.insert(tk.END, line + '\n')
        
        # Make URLs clickable
        text_widget.tag_bind("url", "<Button-1>", lambda e: self._open_url_from_click(e, text_widget))
        text_widget.tag_bind("url", "<Enter>", lambda e: text_widget.config(cursor="hand2"))
        text_widget.tag_bind("url", "<Leave>", lambda e: text_widget.config(cursor=""))
        
        text_widget.config(state=tk.DISABLED)
        
        # Close button
        ttk.Button(main_frame, text="Close", 
                  command=info_dialog.destroy).pack(pady=(10, 0))
        
        # Center the dialog
        info_dialog.update_idletasks()
        width = info_dialog.winfo_width()
        height = info_dialog.winfo_height()
        x = (info_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (info_dialog.winfo_screenheight() // 2) - (height // 2)
        info_dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def _open_url_from_click(self, event, text_widget):
        """Open URL when clicked in text widget"""
        try:
            # Get the index of the click
            index = text_widget.index(f"@{event.x},{event.y}")
            # Get tags at that position
            tags = text_widget.tag_names(index)
            if "url" in tags:
                # Get the range of the URL tag
                range_start = text_widget.tag_prevrange("url", index + "+1c")
                if range_start:
                    url = text_widget.get(*range_start)
                    webbrowser.open(url)
        except Exception as e:
            print(f"Error opening URL: {e}")
