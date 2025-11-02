import json, socket, datetime

def send(level, msg, extra=None):
    payload = {
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "host": socket.gethostname(),
        "level": level,
        "msg": msg,
        "extra": extra or {}
    }
    print("ALERT>", json.dumps(payload))          # TODO: kafka.send('alerts', payload)

if __name__ == "__main__":
    send("warn", "drift detected", {"file":"config.txt"})