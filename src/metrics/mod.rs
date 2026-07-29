use std::sync::OnceLock;

use prometheus::{
    HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    register_histogram_vec, register_int_counter_vec, register_int_gauge, register_int_gauge_vec,
};

const LLM_DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
];

/// Metrics shared by the data plane, lifecycle engine, and observer.
///
/// Labels deliberately exclude service operation IDs and request IDs: both are
/// unbounded and would create an unsafe Prometheus cardinality surface.
#[derive(Clone)]
pub struct RouterMetrics {
    pub requests: IntCounterVec,
    pub request_seconds: HistogramVec,
    pub parked: IntGaugeVec,
    pub parking_seconds: HistogramVec,
    pub transitions: IntCounterVec,
    pub probe_failures: IntCounterVec,
    pub active: IntGaugeVec,
    pub active_streams: IntGaugeVec,
    pub service_state: IntGaugeVec,
    /// Cold-start phase timings. `phase` and `result` are fixed vocabularies;
    /// `service` is bounded by the validated configuration.
    pub start_phase_seconds: HistogramVec,
    pub global_parked: IntGauge,
}

impl RouterMetrics {
    fn register() -> Self {
        Self {
            requests: register_int_counter_vec!(
                Opts::new(
                    "vllm_router_requests_total",
                    "Completed router requests by canonical model and outcome"
                ),
                &["model", "status"]
            )
            .expect("request metric registration"),
            request_seconds: register_histogram_vec!(
                HistogramOpts::new(
                    "vllm_router_request_duration_seconds",
                    "End-to-end downstream request duration"
                )
                .buckets(LLM_DURATION_BUCKETS.to_vec()),
                &["model"]
            )
            .expect("request latency metric registration"),
            parked: register_int_gauge_vec!(
                Opts::new(
                    "vllm_router_parked_requests",
                    "Requests waiting for model readiness"
                ),
                &["model"]
            )
            .expect("parked metric registration"),
            parking_seconds: register_histogram_vec!(
                HistogramOpts::new(
                    "vllm_router_parking_duration_seconds",
                    "Time spent waiting for model readiness"
                )
                .buckets(LLM_DURATION_BUCKETS.to_vec()),
                &["model"]
            )
            .expect("parking latency metric registration"),
            transitions: register_int_counter_vec!(
                Opts::new(
                    "vllm_router_transitions_total",
                    "Lifecycle transition attempts"
                ),
                &["service", "transition", "result"]
            )
            .expect("transition metric registration"),
            probe_failures: register_int_counter_vec!(
                Opts::new(
                    "vllm_router_probe_failures_total",
                    "Failed observation probes"
                ),
                &["service", "probe"]
            )
            .expect("probe metric registration"),
            active: register_int_gauge_vec!(
                Opts::new(
                    "vllm_router_active_requests",
                    "Exact active request count by service"
                ),
                &["service"]
            )
            .expect("active metric registration"),
            active_streams: register_int_gauge_vec!(
                Opts::new(
                    "vllm_router_active_streams",
                    "Exact active streaming response count by service"
                ),
                &["service"]
            )
            .expect("active stream metric registration"),
            service_state: register_int_gauge_vec!(
                Opts::new(
                    "vllm_router_service_state",
                    "One-hot lifecycle state for each service"
                ),
                &["service", "state"]
            )
            .expect("state metric registration"),
            start_phase_seconds: register_histogram_vec!(
                HistogramOpts::new(
                    "vllm_router_start_phase_duration_seconds",
                    "Cold-start duration by phase and outcome"
                )
                .buckets(LLM_DURATION_BUCKETS.to_vec()),
                &["service", "phase", "result"]
            )
            .expect("cold-start phase metric registration"),
            global_parked: register_int_gauge!(
                "vllm_router_parked_requests_total_current",
                "Current parked requests across every model"
            )
            .expect("global parked metric registration"),
        }
    }
}

/// Return the process-wide metrics set.
pub fn global() -> &'static RouterMetrics {
    static METRICS: OnceLock<RouterMetrics> = OnceLock::new();
    METRICS.get_or_init(RouterMetrics::register)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_are_registered_in_the_default_pingora_registry() {
        global().global_parked.set(1);
        assert!(
            prometheus::gather()
                .iter()
                .any(|family| family.get_name() == "vllm_router_parked_requests_total_current")
        );
    }

    #[test]
    fn latency_histograms_use_llm_scale_buckets() {
        global()
            .request_seconds
            .with_label_values(&["bucket-test"])
            .observe(0.01);
        global()
            .parking_seconds
            .with_label_values(&["bucket-test"])
            .observe(120.0);
        global()
            .start_phase_seconds
            .with_label_values(&["bucket-test", "warmup", "success"])
            .observe(10.0);

        for name in [
            "vllm_router_request_duration_seconds",
            "vllm_router_parking_duration_seconds",
            "vllm_router_start_phase_duration_seconds",
        ] {
            let families = prometheus::gather();
            let histogram = families
                .iter()
                .find(|family| family.get_name() == name)
                .expect("histogram should be registered")
                .get_metric()[0]
                .get_histogram()
                .clone();
            let upper_bounds: Vec<_> = histogram
                .get_bucket()
                .iter()
                .map(|bucket| bucket.get_upper_bound())
                .collect();
            assert_eq!(upper_bounds, LLM_DURATION_BUCKETS);
        }
    }
}
