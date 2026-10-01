use anyhow::{Context, Result, bail};
use std::time::Duration;
use tokio::sync::{OnceCell, mpsc, oneshot};

use super::pinn::{get_pinn, pinn_infer_batch};
use super::types::PinnInputs;

pub(crate) struct PinnRequest {
    pub(crate) inputs: Vec<PinnInputs>,
    pub(crate) responder: oneshot::Sender<Result<Vec<[f32; 4]>, String>>,
}

static PINN_QUEUE: OnceCell<mpsc::Sender<PinnRequest>> = OnceCell::const_new();

const PINN_QUEUE_CAPACITY: usize = 1024;
const DYNAMIC_BATCH_WINDOW: Duration = Duration::from_millis(5);
const MAX_DYNAMIC_BATCH_SIZE: usize = 256;

async fn get_pinn_queue() -> mpsc::Sender<PinnRequest> {
    PINN_QUEUE
        .get_or_init(|| async {
            let (tx, mut rx) = mpsc::channel::<PinnRequest>(PINN_QUEUE_CAPACITY);
            tokio::spawn(async move {
                let mut pending = None;
                while let Some(first_req) = match pending.take() {
                    Some(req) => Some(req),
                    None => rx.recv().await,
                } {
                    let mut batch_inputs = first_req.inputs;
                    let mut responders = vec![(batch_inputs.len(), first_req.responder)];

                    let deadline = tokio::time::Instant::now() + DYNAMIC_BATCH_WINDOW;
                    while batch_inputs.len() < MAX_DYNAMIC_BATCH_SIZE {
                        let timeout =
                            deadline.saturating_duration_since(tokio::time::Instant::now());
                        if timeout.is_zero() {
                            break;
                        }
                        tokio::select! {
                            biased;
                            Some(req) = rx.recv() => {
                                let count = req.inputs.len();
                                if batch_inputs.len() + count > MAX_DYNAMIC_BATCH_SIZE {
                                    pending = Some(req);
                                    break;
                                }
                                batch_inputs.extend(req.inputs);
                                responders.push((count, req.responder));
                                if batch_inputs.len() >= MAX_DYNAMIC_BATCH_SIZE {
                                    break;
                                }
                            }
                            _ = tokio::time::sleep(timeout) => {
                                break;
                            }
                        }
                    }

                    let pinn_ref = match get_pinn().await {
                        Ok(pinn) => pinn,
                        Err(err) => {
                            for (_, responder) in responders {
                                let _ =
                                    responder.send(Err(format!("PINN model unavailable: {err:#}")));
                            }
                            continue;
                        }
                    };
                    let outputs = tokio::task::spawn_blocking(move || {
                        pinn_infer_batch(
                            &pinn_ref.model,
                            &pinn_ref.device,
                            &pinn_ref.norm,
                            &batch_inputs,
                        )
                    })
                    .await
                    .map_err(|err| format!("PINN worker failed: {err}"))
                    .and_then(|rows| {
                        if rows.len() != responders.iter().map(|(count, _)| count).sum::<usize>() {
                            Err("PINN worker returned an incomplete batch".to_string())
                        } else {
                            Ok(rows)
                        }
                    });

                    let mut offset = 0;
                    for (count, responder) in responders {
                        let slice = outputs
                            .as_ref()
                            .map(|rows| rows[offset..offset + count].to_vec())
                            .map_err(Clone::clone);
                        offset += count;
                        let _ = responder.send(slice);
                    }
                }
            });
            tx
        })
        .await
        .clone()
}

pub async fn infer_pinn_batch_async(inputs: Vec<PinnInputs>) -> Result<Vec<[f32; 4]>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if inputs.len() > MAX_DYNAMIC_BATCH_SIZE {
        bail!("PINN request exceeds the maximum batch size of {MAX_DYNAMIC_BATCH_SIZE}");
    }
    let queue = get_pinn_queue().await;
    let (tx, rx) = oneshot::channel();
    enqueue_pinn_request(
        &queue,
        PinnRequest {
            inputs,
            responder: tx,
        },
    )?;
    rx.await
        .context("PINN worker stopped without responding")?
        .map_err(anyhow::Error::msg)
}

pub(crate) fn enqueue_pinn_request(
    queue: &mpsc::Sender<PinnRequest>,
    request: PinnRequest,
) -> Result<()> {
    queue.try_send(request).map_err(|err| match err {
        mpsc::error::TrySendError::Full(_) => {
            anyhow::anyhow!("PINN queue is full; retry the request later")
        }
        mpsc::error::TrySendError::Closed(_) => {
            anyhow::anyhow!("PINN worker is unavailable")
        }
    })
}
