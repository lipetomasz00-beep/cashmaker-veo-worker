# Nowy endpoint - zamień stary @app.route('/videos/<path:filename>')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """GET /videos/<filename> - PUBLIC ACCESS (no auth required)"""
    return send_from_directory(STORAGE_DIR, filename)

