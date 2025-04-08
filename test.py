try:
    from app.services.sessionService import getOrCreateSession
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {type(e).__name__}: {e}")
    
    # Show the full traceback
    import traceback
    traceback.print_exc()