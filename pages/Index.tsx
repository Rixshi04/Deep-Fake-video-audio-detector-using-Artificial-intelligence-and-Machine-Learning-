import { Link } from "react-router-dom";

export default function Index() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
      <div className="mx-auto max-w-5xl">
        <p className="mb-3 text-sm font-semibold uppercase tracking-[0.2em] text-cyan-400">
          Research prototype
        </p>
        <h1 className="text-4xl font-bold sm:text-6xl">
          Deepfake Video &amp; Audio Detector
        </h1>
        <p className="mt-6 max-w-2xl text-lg text-slate-300">
          Upload media to the Flask backend for experimental video or audio
          analysis. Results are only available when the corresponding detector
          and trained model artifacts are present.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link className="rounded-lg bg-cyan-500 px-5 py-3 font-semibold text-slate-950" to="/demo">
            Open Demo
          </Link>
          <Link className="rounded-lg border border-slate-700 px-5 py-3 font-semibold" to="/technology">
            Technology
          </Link>
          <Link className="rounded-lg border border-slate-700 px-5 py-3 font-semibold" to="/resources">
            Resources
          </Link>
        </div>
      </div>
    </main>
  );
}
