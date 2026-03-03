import customtkinter as ctk
from tkinter import messagebox

class FakeNewsDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection System")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)
        
        # Set theme and colors
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Color palette
        self.colors = {
            "bg_primary": "#1a1a2e",      # Deep dark blue-gray
            "bg_secondary": "#16213e",     # Slightly lighter blue
            "bg_card": "#0f3460",          # Card background
            "accent_blue": "#e94560",      # Accent color (coral/red)
            "accent_hover": "#ff6b6b",     # Hover state
            "text_primary": "#ffffff",     # White text
            "text_secondary": "#a0a0a0",   # Gray text
            "success": "#00d9ff",          # Cyan for real news
            "danger": "#e94560"            # Red for fake news
        }
        
        # Configure root background
        self.root.configure(fg_color=self.colors["bg_primary"])
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(1, weight=1)  # Main content row expands
        self.root.grid_columnconfigure(0, weight=1)
        
        # Initialize UI components
        self._create_header()
        self._create_main_content()
        self._create_status_bar()
        
        # Analysis state
        self.is_analyzing = False
        
    def _create_header(self):
        """Create the top header section with title and subtitle"""
        self.header_frame = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["bg_secondary"],
            corner_radius=0,
            height=80
        )
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self.header_frame.grid_propagate(False)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Fake News Detection System",
            font=ctk.CTkFont(family="Helvetica", size=28, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.title_label.place(relx=0.5, rely=0.35, anchor="center")
        
        # Subtitle
        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="Real-time NLP Classification",
            font=ctk.CTkFont(family="Helvetica", size=14),
            text_color=self.colors["accent_blue"]
        )
        self.subtitle_label.place(relx=0.5, rely=0.7, anchor="center")
        
    def _create_main_content(self):
        """Create the main content area with left and right panels"""
        self.main_frame = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["bg_primary"],
            corner_radius=0
        )
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure main frame grid
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)  # Left panel takes 3 parts
        self.main_frame.grid_columnconfigure(1, weight=2)  # Right panel takes 2 parts
        
        self._create_left_panel()
        self._create_right_panel()
        
    def _create_left_panel(self):
        """Create left panel with text input"""
        self.left_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.colors["bg_secondary"],
            corner_radius=15,
            border_width=0
        )
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        
        # Input label
        self.input_label = ctk.CTkLabel(
            self.left_frame,
            text="News Article Input",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.input_label.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))
        
        # Text input box with placeholder
        self.text_input = ctk.CTkTextbox(
            self.left_frame,
            fg_color=self.colors["bg_primary"],
            text_color=self.colors["text_primary"],
            font=ctk.CTkFont(family="Helvetica", size=12),
            corner_radius=10,
            border_width=2,
            border_color=self.colors["bg_card"],
            scrollbar_button_color=self.colors["accent_blue"],
            scrollbar_button_hover_color=self.colors["accent_hover"]
        )
        self.text_input.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        # Insert placeholder text
        self.text_input.insert("1.0", "Paste news article here...")
        self.text_input.configure(text_color=self.colors["text_secondary"])
        
        # Bind focus events for placeholder behavior
        self.text_input.bind("<FocusIn>", self._on_text_focus_in)
        self.text_input.bind("<FocusOut>", self._on_text_focus_out)
        
    def _create_right_panel(self):
        """Create right panel with results and controls"""
        self.right_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="transparent",
            corner_radius=0
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        
        # Result Card
        self._create_result_card()
        
        # Control Buttons
        self._create_control_buttons()
        
    def _create_result_card(self):
        """Create the result display card"""
        self.result_card = ctk.CTkFrame(
            self.right_frame,
            fg_color=self.colors["bg_card"],
            corner_radius=15,
            border_width=0
        )
        self.result_card.grid(row=0, column=0, sticky="new", pady=(0, 20))
        self.result_card.grid_columnconfigure(0, weight=1)
        
        # Result header
        self.result_header = ctk.CTkLabel(
            self.result_card,
            text="Analysis Result",
            font=ctk.CTkFont(family="Helvetica", size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        )
        self.result_header.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 15))
        
        # Prediction label
        self.prediction_label = ctk.CTkLabel(
            self.result_card,
            text="Awaiting Analysis",
            font=ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
            text_color=self.colors["text_secondary"]
        )
        self.prediction_label.grid(row=1, column=0, pady=(0, 5))
        
        # Confidence label
        self.confidence_label = ctk.CTkLabel(
            self.result_card,
            text="--%",
            font=ctk.CTkFont(family="Helvetica", size=32, weight="bold"),
            text_color=self.colors["text_secondary"]
        )
        self.confidence_label.grid(row=2, column=0, pady=(0, 10))
        
        # Progress bar
        self.confidence_bar = ctk.CTkProgressBar(
            self.result_card,
            progress_color=self.colors["accent_blue"],
            fg_color=self.colors["bg_secondary"],
            corner_radius=5,
            height=10
        )
        self.confidence_bar.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.confidence_bar.set(0)
        
    def _create_control_buttons(self):
        """Create action buttons"""
        self.button_frame = ctk.CTkFrame(
            self.right_frame,
            fg_color="transparent",
            corner_radius=0
        )
        self.button_frame.grid(row=1, column=0, sticky="sew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        
        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            self.button_frame,
            text="Analyze News",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            fg_color=self.colors["accent_blue"],
            hover_color=self.colors["accent_hover"],
            text_color=self.colors["text_primary"],
            corner_radius=10,
            height=45,
            command=self.analyze_news
        )
        self.analyze_btn.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        # Fetch button
        self.fetch_btn = ctk.CTkButton(
            self.button_frame,
            text="Fetch Live News (API)",
            font=ctk.CTkFont(family="Helvetica", size=14, weight="bold"),
            fg_color=self.colors["bg_secondary"],
            hover_color=self.colors["bg_card"],
            text_color=self.colors["text_primary"],
            border_width=2,
            border_color=self.colors["accent_blue"],
            corner_radius=10,
            height=45,
            command=self.fetch_live_news
        )
        self.fetch_btn.grid(row=1, column=0, sticky="ew")
        
    def _create_status_bar(self):
        """Create bottom status bar"""
        self.status_frame = ctk.CTkFrame(
            self.root,
            fg_color=self.colors["bg_secondary"],
            corner_radius=0,
            height=35
        )
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.status_frame.grid_propagate(False)
        
        # Status message
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Ready",
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color=self.colors["text_secondary"]
        )
        self.status_label.place(x=20, rely=0.5, anchor="w")
        
        # Model accuracy display
        self.accuracy_label = ctk.CTkLabel(
            self.status_frame,
            text="Model Accuracy: 94.2%",
            font=ctk.CTkFont(family="Helvetica", size=11),
            text_color=self.colors["success"]
        )
        self.accuracy_label.place(relx=0.98, rely=0.5, anchor="e")
        
    def _on_text_focus_in(self, event):
        """Handle text box focus in - clear placeholder"""
        current_text = self.text_input.get("1.0", "end-1c")
        if current_text == "Paste news article here...":
            self.text_input.delete("1.0", "end")
            self.text_input.configure(text_color=self.colors["text_primary"])
            
    def _on_text_focus_out(self, event):
        """Handle text box focus out - restore placeholder if empty"""
        current_text = self.text_input.get("1.0", "end-1c").strip()
        if not current_text:
            self.text_input.delete("1.0", "end")
            self.text_input.insert("1.0", "Paste news article here...")
            self.text_input.configure(text_color=self.colors["text_secondary"])
            
    def analyze_news(self):
        """
        Placeholder function for news analysis logic.
        Integrate your NLP model here.
        """
        if self.is_analyzing:
            return
            
        text_content = self.text_input.get("1.0", "end-1c").strip()
        if text_content == "Paste news article here..." or not text_content:
            messagebox.showwarning("Input Required", "Please enter a news article to analyze.")
            return
            
        self.is_analyzing = True
        self.status_label.configure(text="Analyzing...")
        self.analyze_btn.configure(state="disabled", text="Analyzing...")
        
        # Simulate analysis (replace with actual model inference)
        self.root.after(1500, self._complete_analysis)
        
    def _complete_analysis(self):
        """Simulate analysis completion - replace with actual results"""
        import random
        
        # Simulate result (replace with model prediction)
        is_fake = random.choice([True, False])
        confidence = random.uniform(75, 98)
        
        if is_fake:
            self.prediction_label.configure(
                text="FAKE NEWS",
                text_color=self.colors["danger"]
            )
            self.confidence_bar.configure(progress_color=self.colors["danger"])
        else:
            self.prediction_label.configure(
                text="REAL NEWS",
                text_color=self.colors["success"]
            )
            self.confidence_bar.configure(progress_color=self.colors["success"])
            
        self.confidence_label.configure(text=f"{confidence:.1f}%")
        self.confidence_bar.set(confidence / 100)
        self.status_label.configure(text="Analysis complete")
        
        self.is_analyzing = False
        self.analyze_btn.configure(state="normal", text="Analyze News")
        
    def fetch_live_news(self):
        """Placeholder for live news fetching from API"""
        self.status_label.configure(text="Fetching news from API...")
        # TODO: Implement API integration
        messagebox.showinfo("API Integration", "Live news fetching will be implemented here.")
        self.status_label.configure(text="Ready")

# Application entry point
if __name__ == "__main__":
    root = ctk.CTk()
    app = FakeNewsDetectionApp(root)
    root.mainloop()