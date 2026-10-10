import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

from mcp_tools import tools


class SelfAuditingAgent:

    def __init__(self, sensitivity="Medium"):
        self.name = "MCP-Audit-Core-v1"

        # Sensitivity multiplier changes how strictly we score issues
        self.sensitivity_map = {
            "Low": 0.5,
            "Medium": 1.0,
            "High": 1.5
        }

        self.multiplier = self.sensitivity_map.get(
            sensitivity,
            1.0
        )

        print(
            f"[{self.name}] Initialized. Sensitivity: {sensitivity}"
        )

    def run_audit(self):
        """Main audit cycle. Returns a full report dictionary."""

        report = {
            "status": "PASS",
            "issues": [],
            "metrics": {},
            "risk_score": 0,
            "recommendation": "System is healthy. No action required."
        }

        print(f"[{self.name}] Starting Audit...")

        # STEP 1: Check Model Health (Confidence & Accuracy)
        health = tools.get_current_model_health()

        report["metrics"]["health"] = health

        if health.get("avg_confidence", 1.0) < 0.6:
            report["issues"].append(
                "Low Average Confidence detected."
            )

            report["risk_score"] += (
                20 * self.multiplier
            )

        # STEP 2: Check Feature Drift
        drift = tools.check_feature_drift(
            recent_window=50
        )

        report["metrics"]["drift"] = drift

        drift_count = 0

        for feature, data in drift.items():

            if data["is_drifting"]:
                drift_count += 1

                report["issues"].append(
                    f"Major Data Drift in {feature} "
                    f"(Score: {data['drift_score']:.2f})"
                )

        if drift_count > 0:
            report["risk_score"] += (
                drift_count * 15 * self.multiplier
            )

        # STEP 3: Check Failure Rate (Bias Heuristic)
        recents = tools.get_recent_predictions(
            limit=50
        )

        if recents:

            import pandas as pd

            df = pd.DataFrame(recents)

            fail_rate = (
                len(df[df["prediction"] == 1])
                / len(df)
            )

            report["metrics"]["fail_rate"] = fail_rate

            if fail_rate > 0.4:

                report["issues"].append(
                    f"Abnormal Failure Rate "
                    f"({fail_rate:.1%}). Possible Bias."
                )

                report["risk_score"] += (
                    25 * self.multiplier
                )

        report["risk_score"] = int(
            report["risk_score"]
        )

        # Final Status Decision
        if report["risk_score"] > 40:

            report["status"] = "CRITICAL"

            report["recommendation"] = (
                "Immediate Repair Required."
            )

        elif report["risk_score"] > 10:

            report["status"] = "WARNING"

            report["recommendation"] = (
                "Monitor closely. Investigate data quality."
            )

        print(
            f"[{self.name}] Audit Done. "
            f"Status: {report['status']}"
        )

        return report

    def attempt_repair(self):
        """Calls the Repair Kit to fix the detected issues."""

        from mcp_tools import repair_kit

        return repair_kit.perform_auto_repair()