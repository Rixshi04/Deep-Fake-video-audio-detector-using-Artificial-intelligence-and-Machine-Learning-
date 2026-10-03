export default function Technology() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-3xl">
        <a className="text-cyan-400" href="/">← Home</a>
        <h1 className="mt-6 text-4xl font-bold">Technology</h1>
        <ul className="mt-6 space-y-3 text-slate-300">
          <li>React + TypeScript + Vite frontend</li>
          <li>Flask API with asynchronous task polling</li>
          <li>PyTorch CNN architecture for audio experiments</li>
          <li>Librosa and mel-spectrogram preprocessing</li>
          <li>OpenCV-based video handling in the desktop GUI</li>
        </ul>
      </div>
    </main>
  );
}
