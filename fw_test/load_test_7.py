import argparse
import concurrent.futures
import os
import time
from typing import Any

try:
    import httpx
except Exception as e:
    raise SystemExit(
        "httpx is required. Install with: pip install httpx\n"
        f"Import error: {e}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send concurrent /transcribe requests to FW_gpu.py for load testing."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/transcribe", help="API URL")
    parser.add_argument("--audio", required=True, help="Path to test WAV file")
    parser.add_argument("--language", default="ko", help="Language code")
    parser.add_argument("--concurrency", type=int, default=7, help="Concurrent request count")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    return parser.parse_args()


def run_one(
    idx: int,
    url: str,
    audio_bytes: bytes,
    filename: str,
    language: str,
    timeout: float,
    start_at: float,
) -> dict[str, Any]:
    now = time.time()
    if now < start_at:
        time.sleep(start_at - now)

    started = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                files={"file": (filename, audio_bytes, "audio/wav")},
                data={"language": language},
            )
        elapsed = time.time() - started
        if response.status_code != 200:
            return {
                "idx": idx,
                "ok": False,
                "status_code": response.status_code,
                "elapsed_sec": round(elapsed, 3),
                "error": response.text[:300],
            }

        data = response.json()
        return {
            "idx": idx,
            "ok": True,
            "elapsed_sec": round(elapsed, 3),
            "enqueue_seq": data.get("enqueue_seq"),
            "dequeue_seq": data.get("dequeue_seq"),
            "worker_id": data.get("worker_id"),
            "device_index": data.get("device_index"),
            "queue_wait_sec": data.get("queue_wait_sec"),
            "speech_sec": data.get("speech_sec"),
            "text_preview": (data.get("text") or "")[:80],
        }
    except Exception as e:
        elapsed = time.time() - started
        return {
            "idx": idx,
            "ok": False,
            "elapsed_sec": round(elapsed, 3),
            "error": str(e),
        }


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.audio):
        raise SystemExit(f"Audio file not found: {args.audio}")

    with open(args.audio, "rb") as f:
        audio_bytes = f.read()
    filename = os.path.basename(args.audio)

    start_at = time.time() + 1.0
    print(f"Sending {args.concurrency} concurrent requests to: {args.url}")
    print(f"Audio: {args.audio}")
    print("Starting in 1 second...\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [
            ex.submit(
                run_one,
                i + 1,
                args.url,
                audio_bytes,
                filename,
                args.language,
                args.timeout,
                start_at,
            )
            for i in range(args.concurrency)
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x["idx"])
    ok_count = sum(1 for r in results if r["ok"])
    print(f"Done. Success={ok_count}/{len(results)}\n")
    for r in results:
        if r["ok"]:
            print(
                f"[#{r['idx']}] OK "
                f"elapsed={r['elapsed_sec']}s "
                f"enq={r.get('enqueue_seq')} "
                f"deq={r.get('dequeue_seq')} "
                f"worker={r.get('worker_id')} "
                f"gpu={r.get('device_index')} "
                f"queue_wait={r.get('queue_wait_sec')} "
                f"text={repr(r.get('text_preview'))}"
            )
        else:
            print(
                f"[#{r['idx']}] FAIL "
                f"elapsed={r['elapsed_sec']}s "
                f"status={r.get('status_code')} "
                f"error={r.get('error')}"
            )


if __name__ == "__main__":
    main()
