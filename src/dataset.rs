//! Dataset discovery. No downloading happens here — by project rule, the user
//! is asked before anything is fetched.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Which MAESTRO split a file belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Split {
    Train,
    Validation,
    Test,
    Unknown,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
            Split::Unknown => "unknown",
        }
    }

    fn parse(text: &str) -> Self {
        match text.trim().to_lowercase().as_str() {
            "train" => Split::Train,
            "validation" | "val" | "valid" => Split::Validation,
            "test" => Split::Test,
            _ => Split::Unknown,
        }
    }
}

/// Infer the split from the path components (`train`, `validation`, `test`).
pub fn split_of(path: &Path) -> Split {
    for component in path.components() {
        let text = component.as_os_str().to_string_lossy().to_lowercase();
        match text.as_str() {
            "train" => return Split::Train,
            "validation" | "val" | "valid" => return Split::Validation,
            "test" => return Split::Test,
            _ => {}
        }
    }
    Split::Unknown
}

/// Collect `.mid` and `.midi` files below `root`, sorted for reproducibility.
pub fn find_midi_files(root: &Path) -> Result<Vec<PathBuf>> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !root.exists() {
        bail!("path does not exist: {}", root.display());
    }
    let mut files = Vec::new();
    walk(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("cannot read entry in {}", dir.display()))?;
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("cannot stat {}", path.display()))?;
        if file_type.is_dir() {
            walk(&path, out)?;
        } else if is_midi(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_midi(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()),
        Some(ext) if ext == "mid" || ext == "midi"
    )
}

/// One row of a MAESTRO-style manifest (CSV index file).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ManifestEntry {
    #[serde(default)]
    pub split: String,
    #[serde(default)]
    pub midi_filename: String,
    #[serde(default)]
    pub duration: f64,
}

/// The split assignment and durations from a dataset's CSV index.
///
/// MAESTRO stores files in year folders (`2018/...midi`), not in
/// `train/` `validation/` `test/` folders, so the split can only be read from
/// the CSV. Looking it up by file name also works because names are unique.
#[derive(Debug, Clone)]
pub struct Manifest {
    root: PathBuf,
    by_relative: HashMap<String, Split>,
    by_name: HashMap<String, Split>,
    /// Sum of the durations declared in the CSV, in seconds.
    declared_duration: f64,
}

impl Manifest {
    /// Load a CSV index such as `maestro-v3.0.0.csv`.
    pub fn load(csv_path: &Path) -> Result<Self> {
        let root = csv_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let mut reader = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(csv_path)
            .with_context(|| format!("cannot open manifest {}", csv_path.display()))?;

        let mut by_relative = HashMap::new();
        let mut by_name = HashMap::new();
        let mut declared_duration = 0.0;

        for (index, record) in reader.deserialize::<ManifestEntry>().enumerate() {
            let entry = record
                .with_context(|| format!("bad row {} in {}", index + 2, csv_path.display()))?;
            if entry.midi_filename.is_empty() {
                continue;
            }
            let split = Split::parse(&entry.split);
            declared_duration += entry.duration;
            let relative = normalize(&entry.midi_filename);
            if let Some(name) = Path::new(&relative).file_name().and_then(|n| n.to_str()) {
                by_name.insert(name.to_string(), split);
            }
            by_relative.insert(relative, split);
        }

        if by_relative.is_empty() {
            bail!("manifest {} lists no MIDI files", csv_path.display());
        }

        Ok(Self {
            root,
            by_relative,
            by_name,
            declared_duration,
        })
    }

    /// Find the split for a file, by relative path first, then by file name.
    pub fn split_for(&self, path: &Path) -> Split {
        if let Ok(relative) = path.strip_prefix(&self.root) {
            let key = normalize(&relative.to_string_lossy());
            if let Some(split) = self.by_relative.get(&key) {
                return *split;
            }
        }
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if let Some(split) = self.by_name.get(name) {
                return *split;
            }
        }
        Split::Unknown
    }

    pub fn len(&self) -> usize {
        self.by_relative.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_relative.is_empty()
    }

    pub fn declared_duration(&self) -> f64 {
        self.declared_duration
    }
}

/// Look for a single `.csv` index file directly inside `dir`.
pub fn find_manifest(dir: &Path) -> Option<PathBuf> {
    if dir.is_file() {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("csv"))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    if candidates.len() == 1 {
        candidates.pop()
    } else {
        None
    }
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names_are_recognized() {
        assert_eq!(
            split_of(Path::new("data/maestro/train/a.mid")),
            Split::Train
        );
        assert_eq!(
            split_of(Path::new("data/maestro/validation/a.mid")),
            Split::Validation
        );
        assert_eq!(split_of(Path::new("data/maestro/test/a.mid")), Split::Test);
        assert_eq!(split_of(Path::new("data/other/a.mid")), Split::Unknown);
    }

    #[test]
    fn midi_extensions_are_case_insensitive() {
        assert!(is_midi(Path::new("x.mid")));
        assert!(is_midi(Path::new("x.MIDI")));
        assert!(!is_midi(Path::new("x.txt")));
    }

    #[test]
    fn manifest_resolves_splits_and_durations() {
        let dir = std::env::temp_dir().join("wmm-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("index.csv");
        std::fs::write(
            &csv_path,
            "split,midi_filename,duration\ntrain,2018/a.midi,12.5\ntest,2017/b.midi,7.5\n",
        )
        .unwrap();

        let manifest = Manifest::load(&csv_path).expect("load manifest");
        assert_eq!(manifest.len(), 2);
        assert!((manifest.declared_duration() - 20.0).abs() < 1e-9);
        assert_eq!(manifest.split_for(&dir.join("2018/a.midi")), Split::Train);
        // Fallback by file name, for paths outside the manifest root.
        assert_eq!(
            manifest.split_for(Path::new("/elsewhere/b.midi")),
            Split::Test
        );
        assert_eq!(manifest.split_for(Path::new("c.midi")), Split::Unknown);

        let _ = std::fs::remove_file(&csv_path);
    }
}
