"""Run a batch of live incidents against a running lab and summarise them.

    python scripts/lab_batch.py                       # the default calibration set
    python scripts/lab_batch.py cartservice:delay adservice:pause

Each incident is queued through the public API, as a visitor would, and
followed on the event stream until it recovers. Every incident is recorded by
the lab itself in runs/live/incidents.jsonl; this script only prints a table.
Each request carries its own X-Forwarded-For entry so the per-visitor cooldown
doesn't serialise a local batch.
"""

import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:8000"
DEFAULT = ["checkoutservice:cpu", "productcatalogservice:delay", "currencyservice:pause", "paymentservice:loss",
           "recommendationservice:cpu", "adservice:pause", "cartservice:loss", "shippingservice:delay",
           "frontend:cpu"]


def post(path, body, visitor):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "x-forwarded-for": visitor})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def follow(incident_id):
    """Block until the incident recovers; return its final public record."""
    with urllib.request.urlopen(API + "/api/live/events", timeout=1200) as stream:
        for line in stream:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            inc = event.get("incident") or {}
            if event.get("type") == "phase" and inc.get("id") == incident_id:
                print(f"    {inc['phase']}", flush=True)
                if inc["phase"] == "recovered":
                    return inc


def main():
    plan = sys.argv[1:] or DEFAULT
    rows = []
    for n, item in enumerate(plan, 1):
        service, fault = item.split(":")
        queued = post("/api/live/break", {"service": service, "fault": fault}, f"10.0.0.{n}")
        print(f"[{n}/{len(plan)}] {service} {fault} -> incident {queued['id']}", flush=True)
        started = time.time()
        inc = follow(queued["id"])
        v, t = inc["verdict"], inc["times"]
        top = inc["causes"][0]["service"] if inc["causes"] else "-"
        rows.append((service, fault, v.get("detected"), round(t["alarm"] - t["injected"], 1) if "alarm" in t else None,
                     top, v.get("top1"), v.get("top3"), round(time.time() - started)))
    print(f"\n{'service':<24}{'fault':<7}{'alarm':<7}{'detect s':>9}  {'agent #1':<24}{'top1':<6}{'top3':<6}{'total s':>8}")
    for r in rows:
        print(f"{r[0]:<24}{r[1]:<7}{str(r[2]):<7}{str(r[3]):>9}  {r[4]:<24}{str(r[5]):<6}{str(r[6]):<6}{r[7]:>8}")
    detected = [r for r in rows if r[2]]
    print(f"\nalarm fired {len(detected)}/{len(rows)}; top-1 {sum(bool(r[5]) for r in rows)}/{len(rows)}; "
          f"top-3 {sum(bool(r[6]) for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
