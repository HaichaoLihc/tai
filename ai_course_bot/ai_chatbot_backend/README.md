## Setup Instructions

1. install requirements `pip install -r requirements.txt`
2. configure env key, see `.env.example`
3. run server `python main.py`

### Configure SQLite Support

1. Download the appropriate `vector0.dylib` and `vss0.dylib` for your machine's hardware from https://github.com/asg017/sqlite-vss/releases. Place `vector0.dylib` and `vss0.dylib` files into the `ai_course_bot/ai-chatbot-backend/app/core/actions/dist/debug` directory.

2. To enable SQLite support, please set `SQLDB = True` in `ai_course_bot/ai-chatbot-backend/app/core/actions/llama_seletor.py`

3. Make sure `current_dir` under `func llama_selector` is set to the correct file path

4. Running python ai_course_bot/ai-chatbot-backend/main.py and launch http://localhost:8000 in preferred browser

## API Documentation

Detailed API documentation is available in the `docs` directory:

- [Local File API](docs/local_file_api.md) - Documentation for the Local File API endpoints
- [Authentication Guide](docs/authentication.md) - Information about API authentication

## Postman Collections

Two Postman collections are available in the `postman` directory for testing the Local File API:

1. **Basic Collection** (`postman/local_file_postman_collection.json`):

   - Simple collection with basic examples
   - Provides essential requests for each endpoint

2. **Enhanced Collection** (`postman/enhanced_local_file_postman_collection.json`):
   - Comprehensive collection with multiple examples for each endpoint
   - Demonstrates various parameter combinations and edge cases
   - Includes better organization with requests grouped by functionality
   - Recommended for frontend development

To generate or update the Postman collections:

```bash
# Generate Basic Local File API Postman collection
python scripts/generate_local_file_postman_collection.py --output postman/local_file_postman_collection.json

# Generate Enhanced Local File API Postman collection
python scripts/enhanced_generate_local_file_postman_collection.py
```

To use the Postman collections:

1. Import the collection into Postman
2. Set the `apiBaseUrl` variable to your server URL (default: http://localhost:8000)
3. Set the `authToken` variable to your authentication token
