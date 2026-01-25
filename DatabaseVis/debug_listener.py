import socketio
import asyncio

sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("Debug Listener Connected")
    await sio.emit('get_active_models', {})

@sio.event
async def active_models_update(data):
    print(f"[EVENT] active_models_update: {data}")

@sio.event
async def disconnect():
    print("Debug Listener Disconnected")

async def main():
    await sio.connect('http://localhost:8080')
    try:
        await sio.wait()
    except KeyboardInterrupt:
        await sio.disconnect()

if __name__ == '__main__':
    asyncio.run(main())
