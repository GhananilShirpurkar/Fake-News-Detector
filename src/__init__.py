# Load Model
self.model = joblib.load(MODEL_PATH)
self.vectorizer = joblib.load(VECTORIZER_PATH)

# Load API Key
load_dotenv()
self.api_key = os.getenv("NEWS_API_KEY")