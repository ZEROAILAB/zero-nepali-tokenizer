import json
import regex
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "tokenizer" / "zero_ne_tokenizer.json"

HOST = "127.0.0.1"
PORT = 8766
MAX_BODY_SIZE = 100_000

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

SPECIAL_TOKENS = {
    name: tokenizer.token_to_id(name)
    for name in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
}


class TokenizeHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8765")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self.send_json({
                "status": "online",
                "project": "ZERO 100M",
                "tokenizer": "Nepali BPE",
                "vocab_size": tokenizer.get_vocab_size(),
                "special_tokens": SPECIAL_TOKENS
            })
        else:
            self.send_json({"error": "Endpoint not found"}, 404)

    def do_POST(self):
        if self.path != "/api/tokenize":
            self.send_json({"error": "Endpoint not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))

            if length <= 0 or length > MAX_BODY_SIZE:
                self.send_json({"error": "Invalid request size"}, 413)
                return

            payload = json.loads(self.rfile.read(length))
            text = payload.get("text", "")

            if not isinstance(text, str):
                self.send_json({"error": "Text must be a string"}, 400)
                return

            encoded = tokenizer.encode(text)
            ids = encoded.ids
            pieces = encoded.tokens
            offsets = encoded.offsets
            decoded = tokenizer.decode(ids)

            token_details = []

            for i, (piece, token_id, offset) in enumerate(
                zip(pieces, ids, offsets)
            ):
                start, end = offset

                token_details.append({
                    "position": i,
                    "token": piece,
                    "readable": text[start:end],
                    "id": token_id,
                    "offset": [start, end]
                })

            grapheme_details = []

            for cluster in regex.finditer(r"\X", text):
                start, end = cluster.span()

                matching = [
                    {
                        "position": i,
                        "id": token_id,
                        "token": piece
                    }
                    for i, (piece, token_id, (ts, te))
                    in enumerate(zip(pieces, ids, offsets))
                    if ts < end and te > start
                ]

                grapheme_details.append({
                    "text": cluster.group(),
                    "offset": [start, end],
                    "token_positions": [item["position"] for item in matching],
                    "token_ids": [item["id"] for item in matching],
                    "raw_tokens": [item["token"] for item in matching]
                })

            self.send_json({
                "project": "ZERO 100M",
                "input_text": text,
                "character_count": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "token_count": len(ids),
                "tokens": token_details,
                "graphemes": grapheme_details,
                "token_ids": ids,
                "decoded_text": decoded,
                "round_trip_matches": decoded == text,
                "vocab_size": tokenizer.get_vocab_size(),
                "special_tokens": SPECIAL_TOKENS
            })

        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
        except Exception as error:
            self.send_json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        print("[ZERO API]", format % args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), TokenizeHandler)

    print("=" * 45)
    print("ZERO 100M TOKENIZATION API")
    print(f"Server: http://{HOST}:{PORT}")
    print(f"Vocabulary: {tokenizer.get_vocab_size():,}")
    print("Endpoint: POST /api/tokenize")
    print("Status: READY")
    print("=" * 45)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nZERO API stopped.")
        server.server_close()
