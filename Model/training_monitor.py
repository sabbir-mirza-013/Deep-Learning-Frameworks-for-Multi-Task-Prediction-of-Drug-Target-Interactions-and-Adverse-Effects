import socketio
import time
from typing import List, Any

class TrainingMonitor:
    def __init__(self, server_url: str = 'http://localhost:8080', model_name: str = 'MyModel'):
        # Use standard synchronous Client which runs network operations in a background thread
        # This is more robust against blocking CPU tasks (like training loops)
        self.sio = socketio.Client(reconnection=True, reconnection_attempts=5)
        self.server_url = server_url
        self.model_name = model_name
        self.connected = False
        
        # Setup callbacks
        self.sio.on('connect', self._on_connect)
        self.sio.on('disconnect', self._on_disconnect)

    def _on_connect(self):
        print(f"[{self.model_name}] Connected to monitoring server.")
        self.sio.emit('register_model', {
            'model_name': self.model_name,
            'status': 'training'
        })
        self.connected = True

    def _on_disconnect(self):
        print(f"[{self.model_name}] Disconnected from server.")
        self.connected = False

    def connect(self):
        """Connects to the server."""
        try:
            if not self.connected:
                self.sio.connect(self.server_url, wait=True)
        except Exception as e:
            print(f"[{self.model_name}] Connection failed: {e}")

    def log_epoch(self, epoch: int, headers: List[str], data: List[List[Any]], best: bool = False):
        """
        Sends an epoch update to the server.
        """
        if not self.sio.connected:
            print(f"[{self.model_name}] Not connected. Attempting reconnect...")
            self.connect()

        if self.sio.connected:
            payload = {
                "model_name": self.model_name,
                "epoch": epoch,
                "headers": headers,
                "data": data,
                "best": best
            }
            try:
                self.sio.emit('epoch_update', payload)
            except Exception as e:
                print(f"[{self.model_name}] Failed to send update: {e}")

    def finish(self):
        """Disconnects cleanly."""
        if self.sio.connected:
            self.sio.disconnect()
            print(f"[{self.model_name}] Finished.")
