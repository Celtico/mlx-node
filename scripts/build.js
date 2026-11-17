// Builds the addon against the prebuilt MLX that ships in the pip wheel
// (mlx-metal on macOS, mlx[cpu] on Linux). Python is only needed to locate
// that wheel at build time; the built addon runs without Python.
//
//   MLX_PYTHON=/path/to/python npm run build
// Default python: python3 (with MLX installed: pip install mlx==<the version in package.json>).
const { execFileSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const candidates = [
  process.env.MLX_PYTHON,
  'python3',
].filter(Boolean);

let mlxDir;
for (const py of candidates) {
  if (py.includes(path.sep) && !fs.existsSync(py)) continue;
  try {
    const pkg = execFileSync(py, ['-c', 'import mlx, os; print(os.path.dirname(mlx.__file__) if mlx.__file__ else list(mlx.__path__)[0])'], { encoding: 'utf8' }).trim();
    const cmakeDir = path.join(pkg, 'share', 'cmake', 'MLX');
    if (fs.existsSync(cmakeDir)) { mlxDir = cmakeDir; break; }
  } catch { /* try next */ }
}
if (!mlxDir) {
  console.error('No MLX found. Install it (pip install mlx) or set MLX_PYTHON.');
  process.exit(1);
}
console.log(`MLX: ${mlxDir}`);
execFileSync(process.execPath, [require.resolve('cmake-js/bin/cmake-js'), 'compile', `--CDMLX_DIR=${mlxDir}`, ...process.argv.slice(2)], { cwd: root, stdio: 'inherit' });
