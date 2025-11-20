from http.server import BaseHTTPRequestHandler, HTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        print("\n=== 收到 POST 请求 ===")
        print("Path:", self.path)
        print("Headers:", self.headers)
        print("Body:", body.decode())
        print("=====================\n")

        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

server = HTTPServer(('0.0.0.0', 8089), Handler)
print("Mock server running on port 8089")
server.serve_forever()
