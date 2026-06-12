#!/usr/bin/env python3
"""Test the forge WebSocket connection."""
import asyncio
import websockets
import json
import sys

async def monitor_forge(session_id: str):
    """Connect to forge WebSocket and print events."""
    uri = f"ws://localhost:7861/ws/forge/{session_id}"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected! Waiting for events...\n")
            
            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=300)
                    event = json.loads(message)
                    
                    event_type = event.get("type", "unknown")
                    data = event.get("data", {})
                    
                    print(f"[{event_type.upper()}]")
                    
                    if event_type == "iteration":
                        print(f"  Iteration: {data.get('iteration', '?')}")
                        print(f"  Score: {data.get('score', '?')}")
                        print(f"  Images: {len(data.get('images', []))}")
                        if data.get('mutations'):
                            print(f"  Mutations: {', '.join(data['mutations'])}")
                    elif event_type == "converged":
                        print(f"  Converged: {data.get('converged', False)}")
                        print(f"  Final score: {data.get('final_score', '?')}")
                        print(f"  Iterations: {data.get('iterations', '?')}")
                        print("\n✅ Forge completed!")
                        break
                    elif event_type == "error":
                        print(f"  Error: {data.get('message', 'unknown')}")
                        print("\n❌ Forge failed!")
                        break
                    else:
                        print(f"  Data: {json.dumps(data, indent=2)[:200]}")
                    
                    print()
                    
                except asyncio.TimeoutError:
                    print("Timeout waiting for event")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: test_ws.py <session_id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(monitor_forge(session_id))
