import os
import joblib
import requests
import customtkinter as ctk
from tkinter import messagebox
from dotenv import load_dotenv
from utils import clean_text


class FakeNewsDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fake News Detection System")
        self.root.geometry("900x600")
        self.root.minsize(800, 500)

        # Appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv("NEWS_API_KEY")

        # Paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "../models/model.pkl")
        VECTORIZER_PATH = os.path.join(BASE_DIR, "../models/vectorizer.pkl")

        # Load Model
        try:
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()
            return

        self.is_analyzing = False

        self.colors = {
            "bg_primary": "#1a1a2e",
            "bg_secondary": "#16213e",
            "bg_card": "#0f3460",
            "accent_blue": "#e94560",
            "accent_hover": "#ff6b6b",
            "text_primary": "#ffffff",
            "text_secondary": "#a0a0a0",
            "success": "#00d9ff",
            "danger": "#e94560"
        }

        self.root.configure(fg_color=self.colors["bg_primary"])
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self._create_header()
        self._create_main_content()
        self._create_status_bar()

    # ==========================
    # UI
    # ==========================

    def _create_header(self):
        header = ctk.CTkFrame(self.root, fg_color=self.colors["bg_secondary"], height=80)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        ctk.CTkLabel(
            header,
            text="Fake News Detection System",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=self.colors["text_primary"]
        ).place(relx=0.5, rely=0.35, anchor="center")

        ctk.CTkLabel(
            header,
            text="Real-time NLP Classification",
            font=ctk.CTkFont(size=14),
            text_color=self.colors["accent_blue"]
        ).place(relx=0.5, rely=0.7, anchor="center")

    def _create_main_content(self):
        main = ctk.CTkFrame(self.root, fg_color=self.colors["bg_primary"])
        main.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)

        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=2)

        self._create_left_panel(main)
        self._create_right_panel(main)

    def _create_left_panel(self, parent):
        left = ctk.CTkFrame(parent, fg_color=self.colors["bg_secondary"], corner_radius=15)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="News Article Input",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors["text_primary"]
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        self.text_input = ctk.CTkTextbox(
            left,
            fg_color=self.colors["bg_primary"],
            text_color=self.colors["text_primary"],
            corner_radius=10
        )
        self.text_input.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))

    def _create_right_panel(self, parent):
        right = ctk.CTkFrame(parent, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._create_result_card(right)
        self._create_buttons(right)

    def _create_result_card(self, parent):
        card = ctk.CTkFrame(parent, fg_color=self.colors["bg_card"], corner_radius=15)
        card.pack(fill="x", pady=(0, 20))

        self.prediction_label = ctk.CTkLabel(
            card,
            text="Awaiting Analysis",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.colors["text_secondary"]
        )
        self.prediction_label.pack(pady=(25, 5))

        self.confidence_label = ctk.CTkLabel(
            card,
            text="--%",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=self.colors["text_secondary"]
        )
        self.confidence_label.pack(pady=(0, 10))

        self.confidence_bar = ctk.CTkProgressBar(card, height=10)
        self.confidence_bar.pack(fill="x", padx=20, pady=(0, 25))
        self.confidence_bar.set(0)

    def _create_buttons(self, parent):
        self.analyze_btn = ctk.CTkButton(
            parent,
            text="Analyze News",
            command=self.analyze_news,
            height=45
        )
        self.analyze_btn.pack(fill="x", pady=(0, 15))

        self.fetch_btn = ctk.CTkButton(
            parent,
            text="Fetch Live News (API)",
            command=self.fetch_live_news,
            height=45
        )
        self.fetch_btn.pack(fill="x")

    def _create_status_bar(self):
        status = ctk.CTkFrame(self.root, fg_color=self.colors["bg_secondary"], height=35)
        status.grid(row=2, column=0, sticky="ew")
        status.grid_propagate(False)

        self.status_label = ctk.CTkLabel(
            status,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=self.colors["text_secondary"]
        )
        self.status_label.place(x=20, rely=0.5, anchor="w")

    # ==========================
    # MODEL INFERENCE
    # ==========================

    def analyze_news(self):
        text_content = self.text_input.get("1.0", "end-1c").strip()

        if not text_content:
            messagebox.showwarning("Input Required", "Please enter a news article.")
            return

        self.status_label.configure(text="Analyzing...")
        self.analyze_btn.configure(state="disabled")

        try:
            cleaned = clean_text(text_content)
            vector = self.vectorizer.transform([cleaned])

            prediction = self.model.predict(vector)[0]
            probability = self.model.predict_proba(vector)[0].max()

            if prediction == 1:
                label = "FAKE NEWS"
                color = self.colors["danger"]
            else:
                label = "REAL NEWS"
                color = self.colors["success"]

            self.prediction_label.configure(text=label, text_color=color)
            self.confidence_label.configure(text=f"{probability*100:.1f}%")
            self.confidence_bar.set(probability)

            self.status_label.configure(text="Analysis complete")

        except Exception as e:
            messagebox.showerror("Error", f"Inference failed:\n{str(e)}")
            self.status_label.configure(text="Error occurred")

        self.analyze_btn.configure(state="normal")

    # ==========================
    # API INTEGRATION
    # ==========================

    def fetch_live_news(self):
        if not self.api_key:
            messagebox.showerror("API Error", "NEWS_API_KEY not found in .env file.")
            return

        self.status_label.configure(text="Fetching news...")

        try:
            url = "https://newsapi.org/v2/top-headlines"
            params = {
                "country": "us",
                "pageSize": 1,
                "apiKey": self.api_key
            }

            response = requests.get(url, params=params)
            data = response.json()

            if data["status"] != "ok":
                messagebox.showerror("API Error", data.get("message", "Unknown error"))
                return

            article = data["articles"][0]
            content = (article.get("title", "") or "") + " " + (article.get("description", "") or "")

            self.text_input.delete("1.0", "end")
            self.text_input.insert("1.0", content)

            self.status_label.configure(text="News fetched successfully")

        except Exception as e:
            messagebox.showerror("API Error", str(e))
            self.status_label.configure(text="Error fetching news")


if __name__ == "__main__":
    root = ctk.CTk()
    app = FakeNewsDetectionApp(root)
    root.mainloop()