use lunar_structures_testbench::{Job, JobStatus, ReportRecord};
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
    db_path: PathBuf,
}

impl Database {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, rusqlite::Error> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(p)?;

        // WAL mode & performance pragmas
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;",
        )?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: p.to_path_buf(),
        };
        db.init_tables()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self, rusqlite::Error> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: PathBuf::from(":memory:"),
        };
        db.init_tables()?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    fn init_tables(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                spec_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                finished_at INTEGER,
                error TEXT,
                manifest_json TEXT,
                metrics_json TEXT,
                job_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);

            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                baseline_delta_json TEXT,
                report_json TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reports_job_id ON reports(job_id);
            CREATE INDEX IF NOT EXISTS idx_reports_kind ON reports(kind);
            CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at);",
        )?;
        Ok(())
    }

    pub fn upsert_job(
        &self,
        job: &Job,
        manifest_json: Option<&str>,
    ) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock();
        let spec_json = serde_json::to_string(&job.spec).unwrap_or_default();
        let metrics_json = serde_json::to_string(&job.last_metrics).unwrap_or_default();
        let job_json = serde_json::to_string(job).unwrap_or_default();
        let status_str = job.status.tag();
        let created_at = job.created_ms as i64;
        let finished_at = job.finished_ms.map(|ms| ms as i64);

        conn.execute(
            "INSERT INTO jobs (id, spec_json, status, created_at, finished_at, error, manifest_json, metrics_json, job_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(id) DO UPDATE SET
                spec_json = excluded.spec_json,
                status = excluded.status,
                finished_at = excluded.finished_at,
                error = excluded.error,
                manifest_json = COALESCE(excluded.manifest_json, jobs.manifest_json),
                metrics_json = excluded.metrics_json,
                job_json = excluded.job_json;",
            params![
                job.id,
                spec_json,
                status_str,
                created_at,
                finished_at,
                job.error_summary,
                manifest_json,
                metrics_json,
                job_json
            ],
        )?;
        Ok(())
    }

    pub fn get_job(&self, id: &str) -> Result<Option<Job>, rusqlite::Error> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT job_json FROM jobs WHERE id = ?1")?;
        let res: Option<String> = stmt.query_row(params![id], |row| row.get(0)).optional()?;
        Ok(res.and_then(|raw| serde_json::from_str(&raw).ok()))
    }

    pub fn list_jobs(&self) -> Result<Vec<Job>, rusqlite::Error> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT job_json FROM jobs ORDER BY created_at DESC")?;
        let rows = stmt.query_map([], |row| {
            let raw: String = row.get(0)?;
            Ok(raw)
        })?;
        let mut jobs = Vec::new();
        for r in rows {
            if let Ok(raw) = r {
                if let Ok(job) = serde_json::from_str::<Job>(&raw) {
                    jobs.push(job);
                }
            }
        }
        Ok(jobs)
    }

    pub fn mark_interrupted_jobs(&self) -> Result<usize, rusqlite::Error> {
        let conn = self.conn.lock();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Find active jobs that were abruptly terminated by process restart
        let mut stmt =
            conn.prepare("SELECT id, job_json FROM jobs WHERE status IN ('queued', 'running')")?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let raw: String = row.get(1)?;
            Ok((id, raw))
        })?;

        let mut to_update = Vec::new();
        for item in rows.flatten() {
            to_update.push(item);
        }

        let count = to_update.len();
        for (id, raw) in to_update {
            if let Ok(mut job) = serde_json::from_str::<Job>(&raw) {
                job.status = JobStatus::Failed;
                job.error_summary =
                    Some("Interrupted: process was restarted while job was active".into());
                job.finished_ms = Some(now as u64);
                let updated_json = serde_json::to_string(&job).unwrap_or_default();
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error = ?1, finished_at = ?2, job_json = ?3 WHERE id = ?4",
                    params![job.error_summary, now, updated_json, id],
                )?;
            }
        }

        Ok(count)
    }

    pub fn insert_report(&self, report: &ReportRecord) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock();
        let delta_json = report
            .baseline_delta_json
            .as_ref()
            .map(|d| serde_json::to_string(d).unwrap_or_default());
        let report_json = serde_json::to_string(&report.report_json).unwrap_or_default();

        conn.execute(
            "INSERT INTO reports (id, job_id, kind, baseline_delta_json, report_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(id) DO UPDATE SET
                job_id = excluded.job_id,
                kind = excluded.kind,
                baseline_delta_json = excluded.baseline_delta_json,
                report_json = excluded.report_json,
                created_at = excluded.created_at;",
            params![
                report.id,
                report.job_id,
                report.kind,
                delta_json,
                report_json,
                report.created_at as i64
            ],
        )?;
        Ok(())
    }

    pub fn get_report(&self, id: &str) -> Result<Option<ReportRecord>, rusqlite::Error> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, job_id, kind, baseline_delta_json, report_json, created_at
             FROM reports WHERE id = ?1",
        )?;
        let row = stmt
            .query_row(params![id], |row| {
                let id: String = row.get(0)?;
                let job_id: String = row.get(1)?;
                let kind: String = row.get(2)?;
                let delta_raw: Option<String> = row.get(3)?;
                let report_raw: String = row.get(4)?;
                let created_at: i64 = row.get(5)?;
                Ok((id, job_id, kind, delta_raw, report_raw, created_at))
            })
            .optional()?;

        match row {
            Some((id, job_id, kind, delta_raw, report_raw, created_at)) => {
                let baseline_delta_json = delta_raw.and_then(|r| serde_json::from_str(&r).ok());
                let report_json =
                    serde_json::from_str(&report_raw).unwrap_or(serde_json::Value::Null);
                Ok(Some(ReportRecord {
                    id,
                    job_id,
                    kind,
                    baseline_delta_json,
                    report_json,
                    created_at: created_at as u64,
                }))
            }
            None => Ok(None),
        }
    }

    pub fn list_reports(&self) -> Result<Vec<ReportRecord>, rusqlite::Error> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT id, job_id, kind, baseline_delta_json, report_json, created_at
             FROM reports ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let job_id: String = row.get(1)?;
            let kind: String = row.get(2)?;
            let delta_raw: Option<String> = row.get(3)?;
            let report_raw: String = row.get(4)?;
            let created_at: i64 = row.get(5)?;
            Ok((id, job_id, kind, delta_raw, report_raw, created_at))
        })?;

        let mut reports = Vec::new();
        for r in rows.flatten() {
            let (id, job_id, kind, delta_raw, report_raw, created_at) = r;
            let baseline_delta_json = delta_raw.and_then(|raw| serde_json::from_str(&raw).ok());
            let report_json = serde_json::from_str(&report_raw).unwrap_or(serde_json::Value::Null);
            reports.push(ReportRecord {
                id,
                job_id,
                kind,
                baseline_delta_json,
                report_json,
                created_at: created_at as u64,
            });
        }
        Ok(reports)
    }
}
