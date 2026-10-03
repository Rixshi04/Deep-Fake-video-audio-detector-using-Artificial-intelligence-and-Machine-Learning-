import { useEffect, useRef, useState } from "react";
import { checkTaskStatus, pollTaskStatus, uploadAudio, uploadVideo, type TaskStatus } from "../api";

export default function Demo() {
  const [file, setFile] = useState<File | null>(null);
  const [type, setType] = useState<"video" | "audio">("video");
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [message, setMessage] = useState("");
  const cancelPolling = useRef<(() => void) | null>(null);

  useEffect(() => () => cancelPolling.current?.(), []);

  async function submit() {
    if (!file) {
      setMessage("Select a media file first.");
      return;
    }

    setMessage("Uploading...");
    setStatus(null);

    try {
      const task = type === "video"
        ? await uploadVideo(file)
        : await uploadAudio(file);

      setMessage("Processing...");
      cancelPolling.current?.();
      cancelPolling.current = pollTaskStatus(
        task.task_id,
        setStatus,
        (result) => {
          setStatus(result);
          setMessage("Analysis completed.");
        },
        (error) => {
          setMessage(error.message);
          void checkTaskStatus(task.task_id).then(setStatus).catch(() => undefined);
        }
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Upload failed.");
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-3xl">
        <a className="text-cyan-400" href="/">← Home</a>
        <h1 className="mt-6 text-4xl font-bold">Detection Demo</h1>
        <p className="mt-3 text-slate-400">
          This UI reports backend errors honestly when detector modules or model weights are unavailable.
        </p>

        <section className="mt-8 rounded-2xl border border-slate-800 bg-slate-900 p-6">
          <div className="flex gap-2">
            <button onClick={() => setType("video")} className={type === "video" ? "rounded px-4 py-2 bg-cyan-500 text-slate-950" : "rounded px-4 py-2 bg-slate-800"}>Video</button>
            <button onClick={() => setType("audio")} className={type === "audio" ? "rounded px-4 py-2 bg-cyan-500 text-slate-950" : "rounded px-4 py-2 bg-slate-800"}>Audio</button>
          </div>

          <input
            className="mt-6 block w-full rounded-lg border border-slate-700 bg-slate-950 p-3"
            type="file"
            accept={type === "video" ? "video/*" : "audio/*"}
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />

          <button
            className="mt-4 w-full rounded-lg bg-cyan-500 px-5 py-3 font-semibold text-slate-950 disabled:opacity-50"
            onClick={submit}
            disabled={!file}
          >
            Analyze {type}
          </button>

          {message && <p className="mt-5 text-sm text-amber-300">{message}</p>}

          {status && (
            <pre className="mt-5 overflow-auto rounded-lg bg-slate-950 p-4 text-sm text-slate-300">
              {JSON.stringify(status, null, 2)}
            </pre>
          )}
        </section>
      </div>
    </main>
  );
}
