import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
      <div className="mx-auto max-w-xl text-center">
        <h1 className="text-5xl font-bold">404</h1>
        <p className="mt-4 text-slate-400">Page not found.</p>
        <Link className="mt-6 inline-block text-cyan-400" to="/">Return home</Link>
      </div>
    </main>
  );
}
